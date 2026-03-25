"""
Redis setup file
Uploads all model JSON metrics into Redis
and provides functions for Flask to read them
"""

import redis
import json
import logging
from pathlib import Path
from log_code import setup_logging
logger = setup_logging("load_data_redis")
import warnings
warnings.filterwarnings("ignore")

# ================= REDIS CLASS =================
class FoodModelRedis:

    def __init__(self, host="172.25.165.28", port=6379, db=0):
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )

            self.client.ping()
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise


    # -------- Upload JSON --------
    def upload_json(self, key, filepath):
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                logger.error(f"File not found -> {filepath}")
                return

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.client.set(key, json.dumps(data))
            logger.info(f"Uploaded successfully -> {key}")

        except Exception as e:
            logger.error(f"Upload failed for {key}: {e}")


    # -------- Read JSON --------
    def read_json(self, key):
        try:
            raw = self.client.get(key)

            if raw:
                logger.info(f"Data retrieved -> {key}")
                return json.loads(raw)

            logger.warning(f"No data found in Redis for key -> {key}")
            return None

        except Exception as e:
            logger.error(f"Error reading {key}: {e}")
            return None


    # -------- Extract Accuracy --------
    def get_model_accuracy(self, model_key):
        try:
            data = self.read_json(model_key)

            if not data:
                return None

            # VGG16 format
            if "test_accuracy" in data:
                return data["test_accuracy"]

            # ResNet50 format
            if "overall_metrics" in data:
                return data["overall_metrics"]["accuracy"]

            # Custom CNN format
            if "accuracy" in data:
                return data["accuracy"]

            logger.warning(f"Accuracy format not recognized -> {model_key}")
            return None

        except Exception as e:
            logger.error(f"Error extracting accuracy for {model_key}: {e}")
            return None


# ================= UPLOAD EXECUTION =================
if __name__ == "__main__":

    try:
        db = FoodModelRedis()

        # 🔴 YOUR DATASET LOCATION
        BASE = Path(r"D:\\DATA_SCIENCE WITH AI\\Internship\\Task_2_Food")

        # Upload Food Classes
        db.upload_json(
            "food:classes",
            BASE / "Food_Class_Values.json"
        )

        # Upload Custom CNN
        db.upload_json(
            "food:model:cnn",
            BASE / "Custom_Cnn" / "C_CNN_Metrics.json"
        )

        # Upload VGG16
        db.upload_json(
            "food:model:vgg16",
            BASE / "Vgg16" / "VGG16_results.json"
        )

        # Upload ResNet50
        db.upload_json(
            "food:model:resnet50",
            BASE / "ResNet50" / "ResNet50_binary_metrics.json"
        )

        logger.info("All files uploaded to Redis successfully")

    except Exception as e:
        logger.critical(f"Fatal error during Redis setup: {e}")
