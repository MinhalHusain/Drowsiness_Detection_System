import os
import json
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

images_folder = r"model\test\images"
labels_folder = r"test\labels"

def load_labels_and_images(images_folder, labels_folder):
    """
    Load images and their corresponding labels from separate folders.
    
    Args:
        images_folder (str): Path to the folder containing image files
        labels_folder (str): Path to the folder containing label files
    
    Returns:
        tuple: Lists of image paths and corresponding labels
    """
    image_paths = []
    image_labels = []
    
    # Iterate through image files
    for image_file in os.listdir(images_folder):
        if image_file.endswith((".jpg", ".png", ".jpeg")):
            # Construct corresponding label file path
            label_file = os.path.join(labels_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            
            # Check if label file exists
            if os.path.exists(label_file):
                # Read label file
                with open(label_file, 'r') as f:
                    label_content = f.read().strip().split()
                
                # First digit determines the class (0 = awake, 1 = drowsy)
                if label_content:
                    label = int(label_content[0])
                    
                    # Add to lists
                    image_paths.append(os.path.join(images_folder, image_file))
                    image_labels.append(label)
    
    return image_paths, image_labels

# Load images and labels
image_paths, image_labels = load_labels_and_images(images_folder, labels_folder)

# Split dataset into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    image_paths, image_labels, test_size=0.2, random_state=42, stratify=image_labels
)

# Optional: Print class distribution
class_distribution = {
    "Total Images": len(image_paths),
    "Awake": image_labels.count(0),
    "Drowsy": image_labels.count(1),
    "Train Images": len(train_images),
    "Test Images": len(test_images)
}
print("Class Distribution:")
for key, value in class_distribution.items():
    print(f"{key}: {value}")

# Save the processed paths and labels for later use
data = {
    "train_images": train_images,
    "train_labels": train_labels,
    "test_images": test_images,
    "test_labels": test_labels
}

# Save as JSON file
with open('processed_data.json', 'w') as f:
    json.dump(data, f, indent=4)

# Optional: Prepare images for model training
def prepare_images(image_paths, target_size=(224, 224)):
    """
    Load and preprocess images.
    
    Args:
        image_paths (list): List of paths to image files
        target_size (tuple): Desired image size for model input
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    images = []
    for path in image_paths:
        # Open image
        img = Image.open(path)
        
        # Resize image
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        images.append(img_array)
    
    return np.array(images)

# Optional: Prepare training and testing image arrays
train_image_array = prepare_images(train_images)
test_image_array = prepare_images(test_images)

# Save preprocessed image arrays (optional)
np.save('train_images.npy', train_image_array)
np.save('test_images.npy', test_image_array)
np.save('train_labels.npy', np.array(train_labels))
np.save('test_labels.npy', np.array(test_labels))

print("Preprocessing complete. Files saved: processed_data.json, train_images.npy, test_images.npy, train_labels.npy, test_labels.npy")