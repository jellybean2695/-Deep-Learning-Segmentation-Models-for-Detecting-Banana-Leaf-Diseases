import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained models
unet_model = load_model("unet_best_model.h5")
segnet_model = load_model("segnet_best_model.h5")

# Load test images
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# Ensure masks are reshaped correctly
if len(Y_test.shape) == 3:
    Y_test = np.expand_dims(Y_test, axis=-1)

# Select a few test samples
num_samples = 5
indices = np.random.choice(len(X_test), num_samples, replace=False)

# Plot images and predictions
fig, axes = plt.subplots(num_samples, 4, figsize=(12, num_samples * 3))

for i, idx in enumerate(indices):
    img = X_test[idx]
    true_mask = Y_test[idx]
    
    # Predict masks
    unet_pred = unet_model.predict(np.expand_dims(img, axis=0))[0]
    segnet_pred = segnet_model.predict(np.expand_dims(img, axis=0))[0]

    # Convert to binary masks
    unet_pred = (unet_pred > 0.5).astype(np.uint8)
    segnet_pred = (segnet_pred > 0.5).astype(np.uint8)

    # Plot original image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Original Image")
    
    # Plot ground truth mask
    axes[i, 1].imshow(true_mask[:, :, 0], cmap="gray")
    axes[i, 1].set_title("Ground Truth Mask")
    
    # Plot U-Net Prediction
    axes[i, 2].imshow(unet_pred[:, :, 0], cmap="gray")
    axes[i, 2].set_title("U-Net Prediction")
    
    # Plot SegNet Prediction
    axes[i, 3].imshow(segnet_pred[:, :, 0], cmap="gray")
    axes[i, 3].set_title("SegNet Prediction")

# Display the plots
plt.tight_layout()
plt.show()
