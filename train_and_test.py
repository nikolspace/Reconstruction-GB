import os
import glob
import cv2
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, 
    Conv2DTranspose, BatchNormalization, Dropout, Activation, 
    MaxPool2D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

"""
This script provides the complete training and testing pipeline for 
microstructural grain boundary reconstruction as described in:
'From Fragmented to Flawless: A Large-Scale Synthetic Micrograph Library...'
"""

# --- MODEL ARCHITECTURE (Attention U-Net) ---

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, 7, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    """
    Attention Gate to highlight salient features 
    from the skip connections.
    """
    wg = Conv2D(num_filters, 1, padding="same")(g)
    ws = Conv2D(num_filters, 1, padding="same")(s)
    out = Activation("relu")(wg + ws)
    out = Conv2D(num_filters, 1, padding="same")(out)
    out = Activation("sigmoid")(out)
    return out * s

def decoder_block(input_tensor, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    s = attention_gate(x, skip_features, num_filters)
    x = Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def build_attention_unet(input_shape, n_classes=1):
    inputs = Input(input_shape)

    # Encoder (Downsampling)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder (Upsampling with Attention)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)

    return Model(inputs, outputs, name="Attention-U-Net")

# --- DATA UTILITIES ---

def load_data(data_path, size=512, limit=2000):
    """Loads image and mask datasets from specified folders."""
    img_list = sorted(glob.glob(os.path.join(data_path, "micro/*.png")))[:limit]
    mask_list = sorted(glob.glob(os.path.join(data_path, "label/*.png")))[:limit]
    
    X, Y = [], []
    print(f"Loading {len(img_list)} image pairs...")
    
    for i_path, m_path in zip(img_list, mask_list):
        # Load and resize input image
        img = cv2.imread(i_path, 0)
        img = cv2.resize(img, (size, size))
        X.append(img)
        
        # Load and resize ground truth mask
        mask = cv2.imread(m_path, 0)
        mask = cv2.resize(mask, (size, size))
        # Threshold mask to 0 and 1
        mask = np.where(mask > 127, 1, 0)
        Y.append(mask)
        
    X = np.array(X) / 255.0  # Normalize
    X = np.expand_dims(X, axis=3)
    Y = np.array(Y).astype(np.float32)
    
    return X, Y

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention U-Net Training and Testing")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root (e.g., ./data/reconstruction)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000, help="Number of images to use for demo")
    
    args = parser.parse_args()

    # 1. Prepare Data
    X, Y = load_data(args.data, limit=args.limit)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 2. Build Model
    input_shape = (512, 512, 1)
    model = build_attention_unet(input_shape, n_classes=1)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Training
    print("\n--- Starting Training ---")
    checkpoint = ModelCheckpoint("best_reconstruction_model.hdf5", monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        batch_size=args.batch, 
        epochs=args.epochs, 
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # 4. Testing & Evaluation
    print("\n--- Starting Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_thresh = (y_pred > 0.5).astype(np.uint8)

    # Mean IoU Calculation
    iou_metric = MeanIoU(num_classes=2)
    # y_test and y_pred_thresh must have same rank for update_state
    iou_metric.update_state(y_test, y_pred_thresh.squeeze())
    print(f"Mean IoU on Test Set: {iou_metric.result().numpy():.4f}")

    # 5. Visualization of a random test sample
    idx = random.randint(0, len(X_test)-1)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title("Input (Fragmented)")
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    
    plt.subplot(132)
    plt.title("Ground Truth")
    plt.imshow(y_test[idx], cmap='gray')
    
    plt.subplot(133)
    plt.title("Attention U-Net Prediction")
    plt.imshow(y_pred_thresh[idx].squeeze(), cmap='gray')
    
    plt.tight_layout()
    plt.show()
    print("Results visualized. Script complete.")
