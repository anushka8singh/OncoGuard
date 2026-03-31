import os
import random

def reduce_images(folder_path, limit):
    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        return
    
    images = os.listdir(folder_path)
    print("Before:", folder_path, len(images))
    
    random.shuffle(images)
    images_to_delete = images[limit:]
    
    for img in images_to_delete:
        os.remove(os.path.join(folder_path, img))
    
    print("After:", folder_path, len(os.listdir(folder_path)))


breast_path = "data/breast"

# Train folders
reduce_images(os.path.join(breast_path, "train/normal"), 400)
reduce_images(os.path.join(breast_path, "train/tumor/benign"), 300)
reduce_images(os.path.join(breast_path, "train/tumor/malignant"), 300)

# Test folders
reduce_images(os.path.join(breast_path, "test/normal"), 80)
reduce_images(os.path.join(breast_path, "test/tumor/benign"), 60)
reduce_images(os.path.join(breast_path, "test/tumor/malignant"), 60)

print("Breast dataset reduction completed.")