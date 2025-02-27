import os
import shutil
import random

# Define dataset paths (Ensure they exist)
BASE_DIR = os.getcwd()  # Gets current working directory
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Wound_dataset_unsorted")
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test")

TEST_SPLIT = 0.2  # 20% of data goes to test set

# Ensure train & test directories exist
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    # Skip if not a directory
    if not os.path.isdir(category_path):
        continue

    os.makedirs(os.path.join(TRAIN_PATH, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_PATH, category), exist_ok=True)

    # Get all images in category folder
    images = os.listdir(category_path)
    random.shuffle(images)

    # Split into train & test
    test_size = int(len(images) * TEST_SPLIT)
    test_images = images[:test_size]
    train_images = images[test_size:]

    # Move images
    for img in train_images:
        shutil.move(os.path.join(category_path, img), os.path.join(TRAIN_PATH, category, img))

    for img in test_images:
        shutil.move(os.path.join(category_path, img), os.path.join(TEST_PATH, category, img))

print("Dataset split complete!")
