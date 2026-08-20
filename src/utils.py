import cv2
import numpy as np
import time
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

IMG_SIZE = (224, 224)

def extract_stream_a_features(image_rgb):
    """
    Extracts 104 Handcrafted Descriptors (Stream A):
      - 80-bin HSV and LAB normalized color histograms
      - 9 color moments (Mean, Std, Skewness per channel)
      - 10 GLCM texture metrics (Contrast, Dissimilarity, Homogeneity, Energy, Correlation)
      - 5 contour shape metrics (Area, Perimeter, Aspect Ratio, Solidity, Extent)
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahe_rgb = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
    
    gray = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(mask) == 0:
        mask = cv2.bitwise_not(mask)
        
    hsv = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [16], [0, 256])
    cv2.normalize(h_hist, h_hist); cv2.normalize(s_hist, s_hist); cv2.normalize(v_hist, v_hist)
    
    a_hist = cv2.calcHist([lab], [1], mask, [16], [0, 256])
    b_hist = cv2.calcHist([lab], [2], mask, [16], [0, 256])
    cv2.normalize(a_hist, a_hist); cv2.normalize(b_hist, b_hist)
    
    color_hists = np.hstack([h_hist.flatten(), s_hist.flatten(), v_hist.flatten(), a_hist.flatten(), b_hist.flatten()])
    
    color_moments = []
    for i in range(3):
        ch = clahe_rgb[:, :, i][mask > 0]
        if len(ch) == 0:
            ch = clahe_rgb[:, :, i].flatten()
        mean_val = np.mean(ch)
        std_val = np.std(ch)
        skew_val = np.mean(((ch - mean_val) / (std_val + 1e-6)) ** 3)
        color_moments.extend([mean_val, std_val, skew_val])
    color_moments = np.array(color_moments, dtype=np.float32)
    
    gray_quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(gray_quantized, distances=[1, 2], angles=[0, np.pi/4], levels=32, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    
    contrast_std = np.std(graycoprops(glcm, 'contrast'))
    dissim_std = np.std(graycoprops(glcm, 'dissimilarity'))
    homog_std = np.std(graycoprops(glcm, 'homogeneity'))
    energy_std = np.std(graycoprops(glcm, 'energy'))
    corr_std = np.std(graycoprops(glcm, 'correlation'))
    
    texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation,
                                 contrast_std, dissim_std, homog_std, energy_std, corr_std], dtype=np.float32)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 1.0
        extent = float(area) / (w * h) if (w * h) > 0 else 1.0
    else:
        area, perimeter, aspect_ratio, solidity, extent = 0.0, 0.0, 1.0, 1.0, 1.0
        
    shape_features = np.array([area, perimeter, aspect_ratio, solidity, extent], dtype=np.float32)
    
    return np.hstack([color_hists, color_moments, texture_features, shape_features])

def evaluate_model_metrics(y_true, preds_prob, start_time, len_test):
    elapsed = (time.time() - start_time) * 1000.0 / len_test
    preds = np.argmax(preds_prob, axis=1) if preds_prob.ndim > 1 else preds_prob
    
    acc = accuracy_score(y_true, preds) * 100.0
    prec = precision_score(y_true, preds, average='macro', zero_division=0)
    rec = recall_score(y_true, preds, average='macro', zero_division=0)
    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    
    return acc, prec, rec, f1, elapsed, preds

def run_mcnemar_test(y_true, preds_baseline, preds_proposed):
    """ Performs McNemar's paired statistical significance test. """
    from scipy import stats
    correct_b = (preds_baseline == y_true)
    correct_p = (preds_proposed == y_true)
    
    b_only = np.sum(correct_b & ~correct_p)
    c_only = np.sum(~correct_b & correct_p)
    
    if (b_only + c_only) == 0:
        p_val = 1.0
    else:
        stat = (abs(b_only - c_only) - 1)**2 / (b_only + c_only)
        p_val = stats.chi2.sf(stat, 1)
        
    return b_only, c_only, p_val

def save_confusion_matrix_plot(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """ Saves a 300 DPI heatmap of the confusion matrix. """
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[c.replace('_', '\n') for c in class_names],
                yticklabels=[c.replace('_', '\n') for c in class_names])
    plt.title('Test Set Confusion Matrix - PhytoGATE Hybrid', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to {save_path}")
