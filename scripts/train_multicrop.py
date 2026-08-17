"""
PhytoGATE Benchmark Runner: Multi-Crop Foliar Benchmark (Corn, Grape, Peach - 4,000 Images)
"""

import os
import sys
import time
import zipfile
import urllib.request
import gc
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import extract_stream_a_features, evaluate_model_metrics
from src.phytogate import build_phytogate_model

CLASS_NAMES = [
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Peach___Bacterial_spot"
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)

def load_multicrop_dataset(dataset_dir="./multicrop_leaf_dataset"):
    print("[+] Loading Multi-Crop Foliar Dataset...")
    images, labels = [], []
    zip_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_target_path = "./mc_temp.zip"
    
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        os.makedirs(dataset_dir, exist_ok=True)
        urllib.request.urlretrieve(zip_url, zip_target_path)
        with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        if os.path.exists(zip_target_path):
            os.remove(zip_target_path)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_files = []
        name_lower = class_name.lower()
        alt_name = class_name.replace('___', '_').lower()
        
        for root, _, files in os.walk(dataset_dir):
            folder_lower = os.path.basename(root).lower()
            if folder_lower == name_lower or folder_lower == alt_name or (name_lower in folder_lower) or (alt_name in folder_lower):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        class_files.append(os.path.join(root, f))
                        if len(class_files) >= 800: break
                if len(class_files) >= 800: break

        def _read_img(p):
            img = cv2.imread(p)
            if img is not None:
                return cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded = list(executor.map(_read_img, class_files))
            
        valid = [img for img in loaded if img is not None]
        images.extend(valid)
        labels.extend([class_idx] * len(valid))
        print(f"  └─ Loaded [{class_name}]: {len(valid)} images.")
        
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)

def main():
    print("==================================================================")
    print("     PhytoGATE BENCHMARK SUITE: Multi-Crop Foliar (Corn/Grape/Peach)")
    print("==================================================================")
    images_rgb, labels = load_multicrop_dataset()

    print("\n[+] Extracting Stream A Handcrafted Features...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        raw_features = np.array(list(executor.map(extract_stream_a_features, images_rgb)), dtype=np.float32)

    SEEDS = [42, 52, 62]
    results = []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        gc.collect()
        np.random.seed(seed)
        tf.random.set_seed(seed)

        indices = np.arange(len(images_rgb))
        idx_train, idx_temp, y_train, y_temp = train_test_split(indices, labels, test_size=0.30, random_state=seed, stratify=labels)
        idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)

        scaler = StandardScaler()
        X_train_hc = scaler.fit_transform(raw_features[idx_train])
        X_val_hc = scaler.transform(raw_features[idx_val])
        X_test_hc = scaler.transform(raw_features[idx_test])

        pca = PCA(n_components=0.98, random_state=seed)
        X_train_pca = pca.fit_transform(X_train_hc)
        X_val_pca = pca.transform(X_val_hc)
        X_test_pca = pca.transform(X_test_hc)

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train), weights))

        model, b1, b2 = build_phytogate_model(pca_dim=X_train_pca.shape[1], num_classes=NUM_CLASSES)

        model.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit([images_rgb[idx_train], X_train_pca], y_train, validation_data=([images_rgb[idx_val], X_val_pca], y_val), epochs=5, batch_size=32, class_weight=class_weights, verbose=0)

        for b in [b1, b2]:
            b.trainable = True
            for l in b.layers[:-60]: l.trainable = False
            for l in b.layers[-60:]: l.trainable = True

        model.compile(optimizer=optimizers.Adam(2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cb = [callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6),
              callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)]

        model.fit([images_rgb[idx_train], X_train_pca], y_train, validation_data=([images_rgb[idx_val], X_val_pca], y_val), epochs=20, batch_size=32, class_weight=class_weights, callbacks=cb, verbose=0)

        t0 = time.time()
        preds_prob = model.predict([images_rgb[idx_test], X_test_pca], verbose=0)
        acc, prec, rec, f1, lat, _ = evaluate_model_metrics(y_test, preds_prob, t0, len(y_test))
        results.append([acc, prec, rec, f1, lat])
        print(f"[OK] Seed {seed} Multi-Crop Test Accuracy: {acc:.2f}% (F1: {f1:.4f})")

    arr = np.array(results)
    print("\n==================================================================")
    print(f"PhytoGATE Multi-Crop Mean Accuracy: {np.mean(arr[:,0]):.2f}% ± {np.std(arr[:,0]):.2f}% (Peak: {np.max(arr[:,0]):.2f}%)")
    print("==================================================================")

if __name__ == "__main__":
    main()
