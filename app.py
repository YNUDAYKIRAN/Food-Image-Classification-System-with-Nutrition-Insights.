import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import redis
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


# ================= PATH CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ================= REDIS CONNECTION =================
REDIS_HOST = "172.25.165.28"

try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=6379,
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    r.ping()
    print(f"✅ Connected to Redis at {REDIS_HOST}")

except Exception as e:
    print("⚠ Redis not reachable → Offline mode:", e)
    r = None


# ================= LOAD CLASSES =================
class_names = ["pizza","burger","biryani","dosa"]

if r:
    try:
        raw = r.get("food:classes")
        if raw:
            food_classes = json.loads(raw)
            class_names = list(food_classes.keys())
            print("📦 Classes loaded from Redis")
        else:
            print("⚠ food:classes missing in Redis")
    except Exception as e:
        print("⚠ Failed loading classes:", e)


# ================= MODEL PATHS =================
MODEL_PATHS = {
    "cnn": os.path.join(BASE_DIR, "Custom_Cnn", "C_CNN_model.keras"),
    "resnet": os.path.join(BASE_DIR, "ResNet50", "ResNet50_model.keras"),
    "vgg16": os.path.join(BASE_DIR, "Vgg16", "VGG16_food_model.keras")
}

MODEL_INPUT_SIZES = {
    "cnn": (256,256),
    "resnet": (256,256),
    "vgg16": (256,256)
}

models_cache = {}


# ================= UTIL =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS


# ================= LOAD MODEL =================
def get_model(name):

    if name not in MODEL_PATHS:
        raise ValueError("Invalid model selected")

    if name in models_cache:
        return models_cache[name]

    print(f"🔄 Loading model: {name}")

    models_cache[name] = tf.keras.models.load_model(
        MODEL_PATHS[name],
        compile=False
    )

    print(f"✅ {name} loaded")
    return models_cache[name]


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html", classes=class_names)


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")
        selected_model = request.form.get("model")
        actual_class = request.form.get("actual_class")

        if not file or file.filename == "":
            return jsonify({"error":"No image uploaded"})

        if not allowed_file(file.filename):
            return jsonify({"error":"Invalid file type. Upload JPG/PNG only."})

        if selected_model not in MODEL_PATHS:
            return jsonify({"error":"Invalid model selected"})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=MODEL_INPUT_SIZES[selected_model])
        img_array = image.img_to_array(img)

        if selected_model == "vgg16":
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        model = get_model(selected_model)
        preds = model.predict(img_array)

        index = int(np.argmax(preds))
        confidence = round(float(np.max(preds))*100,2)

        if index >= len(class_names):
            return jsonify({"error":"Prediction index out of range"})

        predicted_class = class_names[index]

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "actual_class": actual_class,
            "model_name": selected_model,
            "image_path": "/static/uploads/"+filename
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error":str(e)})


# ================= FOOD DETAILS =================
@app.route("/food_details/<food>")
def food_details(food):

    if not r:
        return jsonify({"error":"Redis not connected"})

    raw = r.get("food:classes")

    if not raw:
        return jsonify({"error":"No food data in Redis"})

    try:
        foods = json.loads(raw)
    except Exception:
        return jsonify({"error":"Corrupted JSON in Redis"})

    result = foods.get(food.lower())

    if not result:
        return jsonify({"error":"food not found"})

    return jsonify(result)


# ================= MODEL METRICS =================
@app.route("/model_metrics/<model>")
def model_metrics(model):

    if not r:
        return jsonify({"error":"Redis not connected"})

    key_map = {
        "cnn":"food:model:cnn",
        "vgg16":"food:model:vgg16",
        "resnet":"food:model:resnet50"
    }

    key = key_map.get(model)

    if not key:
        return jsonify({"error":"invalid model"})

    raw = r.get(key)

    if not raw:
        return jsonify({"error":"metrics not found in redis"})

    try:
        return jsonify(json.loads(raw))
    except:
        return jsonify({"error":"metrics corrupted"})


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)