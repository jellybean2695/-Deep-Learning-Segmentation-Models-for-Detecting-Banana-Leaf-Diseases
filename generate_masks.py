import os
import cv2
import numpy as np

# Updated Paths
IMAGE_DIR = r"C:\Users\Admin\Desktop\data set\OriginalSet"  # Path to original images
MASK_DIR = r"C:\Users\Admin\Desktop\mask"  # Path to store generated masks

# Ensure the masks folder exists
os.makedirs(MASK_DIR, exist_ok=True)

# Define thresholding function
def generate_mask(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    
    # Adaptive thresholding to detect disease regions
    _, mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # Save mask
    cv2.imwrite(output_path, mask)

# Process each category
categories = ["sigatoka", "cordana", "pestalotiopsis", "healthy"]

for category in categories:
    category_path = os.path.join(IMAGE_DIR, category)
    mask_category_path = os.path.join(MASK_DIR, category)

    # Ensure mask subfolder exists
    os.makedirs(mask_category_path, exist_ok=True)

    # Check if category folder exists
    if not os.path.exists(category_path):
        print(f"Error: Folder '{category_path}' does not exist!")
        continue  # Skip if missing

    # Process images
    for filename in os.listdir(category_path):
        img_path = os.path.join(category_path, filename)
        mask_path = os.path.join(mask_category_path, filename)

        generate_mask(img_path, mask_path)

print("Mask generation completed!")
