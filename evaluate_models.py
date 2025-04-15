import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import jaccard_score

# Load trained models
unet_model = load_model("unet_best_model.h5")
segnet_model = load_model("segnet_best_model.h5")

# Load test dataset
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# Ensure masks are reshaped correctly
if len(Y_test.shape) == 3:
    Y_test = np.expand_dims(Y_test, axis=-1)

# Ensure Ground Truth (Y_test) is binary (0 or 1)
Y_test = (Y_test > 0.5).astype(np.uint8)

# Function to calculate IoU and Dice Score
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)  # Continuous outputs (0 to 1)
    Y_pred = (Y_pred > 0.5).astype(np.uint8)  # Convert to binary (0 or 1)

    iou_scores = []
    dice_scores = []

    for i in range(len(Y_test)):
        y_true_flat = Y_test[i].flatten()
        y_pred_flat = Y_pred[i].flatten()

        # Ensure y_true and y_pred contain only 0s and 1s
        if np.sum(y_true_flat) == 0 and np.sum(y_pred_flat) == 0:
            iou = 1.0  # Perfect match if both are empty
            dice = 1.0
        else:
            iou = jaccard_score(y_true_flat, y_pred_flat, average='binary')
            dice = (2.0 * np.sum(y_true_flat * y_pred_flat)) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-7)

        iou_scores.append(iou)
        dice_scores.append(dice)

    return np.mean(iou_scores), np.mean(dice_scores)

# Evaluate U-Net
iou_unet, dice_unet = evaluate_model(unet_model, X_test, Y_test)
print(f"U-Net Evaluation -> IoU: {iou_unet:.4f}, Dice Score: {dice_unet:.4f}")

# Evaluate SegNet
iou_segnet, dice_segnet = evaluate_model(segnet_model, X_test, Y_test)
print(f"SegNet Evaluation -> IoU: {iou_segnet:.4f}, Dice Score: {dice_segnet:.4f}")
