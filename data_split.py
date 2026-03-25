import os
import random
import shutil

# ==========================
# CONFIG (WINDOWS PATHS)
# ==========================
SOURCE_DIR = "D:\\DATA_SCIENCE WITH AI\\Internship\\Task_2_Food\\Dataset\\Food Classification dataset"
DEST_DIR = "D:\\DATA_SCIENCE WITH AI\\Internship\\Task_2_Food\\Dataset\\Food_Class"

TRAIN_COUNT = 200
VAL_COUNT = 50
TEST_COUNT = 10

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

# ==========================
# CREATE OUTPUT FOLDERS
# ==========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# ==========================
# SPLIT DATA
# ==========================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    required = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    if len(images) < required:
        print(f"⚠️ Skipping {class_name} (only {len(images)} images)")
        continue

    random.shuffle(images)

    train_imgs = images[:TRAIN_COUNT]
    val_imgs = images[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_imgs = images[TRAIN_COUNT + VAL_COUNT:required]

    # Create class directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(DEST_DIR, "train", class_name, img))

    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(DEST_DIR, "val", class_name, img))

    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(DEST_DIR, "test", class_name, img))

    print(f"✅ {class_name}: Train={len(train_imgs)}, "
          f"Val={len(val_imgs)}, Test={len(test_imgs)}")

print("\n🎉 Dataset split completed!")