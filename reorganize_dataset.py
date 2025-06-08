import os
import shutil

# Source and destination paths
source_dir = "animaldaset/animals/animals"
dest_dir = "animaldaset/train"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Move each class folder to the train directory
for class_name in os.listdir(source_dir):
    source_path = os.path.join(source_dir, class_name)
    dest_path = os.path.join(dest_dir, class_name)
    
    if os.path.isdir(source_path):
        # If destination already exists, remove it first
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        # Move the directory
        shutil.move(source_path, dest_path)
        print(f"Moved {class_name} to train directory")

print("Dataset reorganization complete!") 