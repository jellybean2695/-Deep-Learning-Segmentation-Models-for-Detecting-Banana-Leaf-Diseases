import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X.npy")
Y = np.load("Y.npy")

# Ensure Y has the correct shape
if len(Y.shape) == 3:  # If missing channel dimension
    Y = np.expand_dims(Y, axis=-1)

# Split dataset: 80% Train, 20% Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Save datasets
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

print("Dataset successfully split and saved!")
