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


brain_path = "data/brain"

# Train folders
reduce_images(os.path.join(brain_path, "train/normal"), 500)
reduce_images(os.path.join(brain_path, "train/tumor/glioma_tumor"), 300)
reduce_images(os.path.join(brain_path, "train/tumor/meningioma_tumor"), 300)
reduce_images(os.path.join(brain_path, "train/tumor/pituitary_tumor"), 300)

# Test folders
reduce_images(os.path.join(brain_path, "test/normal"), 100)
reduce_images(os.path.join(brain_path, "test/tumor/glioma_tumor"), 60)
reduce_images(os.path.join(brain_path, "test/tumor/meningioma_tumor"), 60)
reduce_images(os.path.join(brain_path, "test/tumor/pituitary_tumor"), 60)

print("Dataset reduction completed.")