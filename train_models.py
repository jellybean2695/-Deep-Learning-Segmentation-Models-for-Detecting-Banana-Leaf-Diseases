import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_model import build_unet
from segnet_model import build_segnet

# Load Preprocessed Data
X = np.load("X.npy")
Y = np.load("Y.npy")

# Ensure Y has the correct shape for training
if len(Y.shape) == 3:  # If missing channel dimension
    Y = np.expand_dims(Y, axis=-1)

# Train U-Net
print("Training U-Net...")
unet_model = build_unet()
unet_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # **Compile Here**
unet_checkpoint = ModelCheckpoint("unet_best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
unet_model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=8, callbacks=[unet_checkpoint])

# Train SegNet
print("Training SegNet...")
segnet_model = build_segnet()
segnet_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # **Compile Here**
segnet_checkpoint = ModelCheckpoint("segnet_best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
segnet_model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=8, callbacks=[segnet_checkpoint])

print("Training completed!")
