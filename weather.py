import os
import matplotlib.pyplot as plt
from collections import Counter

# Root folder where class folders are stored
dataset_path = "C:\\Users\\obada\\.cache\\kagglehub\\datasets\\pratik2901\\multiclass-weather-dataset\\versions\\3\\Multi-class Weather Dataset"

# Step 1: Get class names (folder names)
class_names = os.listdir(dataset_path)
print("Classes:", class_names)

# Step 2: Count number of images in each class
class_counts = {}
for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    image_files = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[class_name] = len(image_files)

# Step 3: Print counts
print("\nImage counts per class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

# Step 4: Plot class distribution
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.title('Class Distribution of Weather Images')
plt.xlabel('Weather Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
