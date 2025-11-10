#Nama : Zaky Askar Sonia NIM : 4212301088
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, f1_score, recall_score)
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
import pickle 

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = r'/home/zaky/Documents/KuliahD4/MachineVision/ATS/archive'
FILE_PATH = os.path.join(DATA_DIR, 'emnist-letters-train.csv')
OUTPUT_DIR = r'/home/zaky/Documents/KuliahD4/MachineVision/ATS/archive/Results1'
LOG_FILE = os.path.join(OUTPUT_DIR, 'evaluation_log.txt')

SAMPLES_PER_CLASS = 500
TOTAL_SAMPLES = 26 * SAMPLES_PER_CLASS
IMAGE_SIZE = 28

RUN_TUNING_MODE = True  
RUN_LOOCV_FINAL = True 

HOG_PARAMS_FINAL = {'orientations': 9, 'ppc': (8, 8), 'cpb': (2, 2)}
SVM_PARAMS_FINAL = {'C': 10.0, 'kernel': 'linear'}

def write_log(message):
    print(message)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(message + '\n')

def load_and_sample_data(file_path):
    write_log("Memuat dan melakukan sampling data seimbang...")
    df = pd.read_csv(file_path, header=None)
    X_full = df.iloc[:, 1:].values.astype('float32') / 255.0
    y_full = df.iloc[:, 0].values - 1  

    X_sampled_list, y_sampled_list = [], []

    for class_label in range(26):
        class_indices = np.where(y_full == class_label)[0]
        if len(class_indices) >= SAMPLES_PER_CLASS:
            selected_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)
        else:
            selected_indices = class_indices
        X_sampled_list.append(X_full[selected_indices])
        y_sampled_list.append(y_full[selected_indices])

    X_sampled = np.concatenate(X_sampled_list, axis=0)
    y_sampled = np.concatenate(y_sampled_list, axis=0)

    write_log(f"Total sampel final: {len(X_sampled)} ({len(np.unique(y_sampled))} kelas)")
    return X_sampled, y_sampled

def extract_hog_features(images, orientations, ppc, cpb):
    hog_features = []
    write_log(f"\n[HOG] Ekstraksi fitur: Orient={orientations}, PPC={ppc}, CPB={cpb}...")

    for image in tqdm(images, desc="Ekstraksi HOG"):
        image_2d = image.reshape(IMAGE_SIZE, IMAGE_SIZE)
        features = hog(image_2d,
                       orientations=orientations,
                       pixels_per_cell=ppc,
                       cells_per_block=cpb,
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       feature_vector=True)
        hog_features.append(features)

    X_features = np.array(hog_features)
    write_log(f"Dimensi fitur HOG: {X_features.shape}")
    return X_features

def tune_parameters(X_raw_images, y_labels):
    write_log("\n---------- FASE: PENYETELAN PARAMETER ----------")
    
    svm_param_grid = [
        {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
    ]

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    sample_size = min(3000, len(X_raw_images))
    indices = np.random.choice(len(X_raw_images), sample_size, replace=False)
    X_raw_tune = X_raw_images[indices]
    y_tune = y_labels[indices]

    X_features_default_tune = extract_hog_features(X_raw_tune, **HOG_PARAMS_FINAL)

    svm_model = SVC(random_state=42, max_iter=30000)
    svm_grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=svm_param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    write_log("Memulai tuning SVM pada fitur HOG default...")
    start_time = time.time()
    svm_grid_search.fit(X_features_default_tune, y_tune)
    end_time = time.time()
    
    best_svm_params = svm_grid_search.best_params_
    best_svm_score = svm_grid_search.best_score_
    
    write_log(f"Waktu tuning SVM: {(end_time - start_time):.2f} detik")
    write_log(f"Parameter SVM Terbaik (pada HOG default): {best_svm_params}")
    write_log(f"Skor Validasi Silang Terbaik (SVM): {best_svm_score:.4f}")

    write_log("\nMencoba beberapa kombinasi HOG umum dengan SVM terbaik...")
    best_overall_score = best_svm_score
    best_overall_hog_params = HOG_PARAMS_FINAL 
    best_overall_svm_params = best_svm_params
    
    hog_options = [
        {'orientations': 8, 'ppc': (8, 8), 'cpb': (2, 2)},
        {'orientations': 9, 'ppc': (8, 8), 'cpb': (2, 2)},
        {'orientations': 9, 'ppc': (8, 8), 'cpb': (3, 3)},
        {'orientations': 10, 'ppc': (8, 8), 'cpb': (2, 2)},
        {'orientations': 10, 'ppc': (8, 8), 'cpb': (3, 3)},
    ]

    for hog_params in hog_options:
        write_log(f"Mencoba HOG params: {hog_params}")

        X_features_hog_tune = extract_hog_features(X_raw_tune, **hog_params)
        
        temp_svm_model = SVC(**best_svm_params, random_state=42, max_iter=30000)
        temp_scores = cross_val_score(temp_svm_model, X_features_hog_tune, y_tune, cv=inner_cv, scoring='accuracy', n_jobs=-1, verbose=0)
        temp_score = temp_scores.mean()
        
        if temp_score > best_overall_score:
            best_overall_score = temp_score
            best_overall_hog_params = hog_params

            write_log(f" -> Memperbarui Best HOG: {hog_params}, Score: {temp_score:.4f}")

    write_log(f"\nParameter Terbaik Secara Keseluruhan:")
    write_log(f" - HOG: {best_overall_hog_params}")
    write_log(f" - SVM: {best_overall_svm_params}")
    write_log(f" - Skor CV Keseluruhan: {best_overall_score:.4f}")

    return best_overall_hog_params, best_overall_svm_params


def evaluate_loocv_with_metrics(X_features, y_labels, svm_params):
    write_log(f"\n---------- FASE: EVALUASI LOOCV FINAL ----------")
    
    model_svm = SVC(C=svm_params['C'], kernel=svm_params['kernel'], 
                    gamma=svm_params.get('gamma', 'scale'), 
                    random_state=42, max_iter=30000)
    
    loocv = LeaveOneOut()

    write_log(f"[LOOCV] Memulai evaluasi akhir (Kernel={svm_params['kernel']}, C={svm_params['C']})...")
    write_log(f"PERINGATAN: LOOCV pada {len(X_features)} sampel akan memakan waktu lama!")
    start_time = time.time()

    y_pred = cross_val_predict(model_svm, X_features, y_labels, cv=loocv, n_jobs=-1, verbose=1)

    end_time = time.time()
    total_hours = (end_time - start_time) / 3600

    acc = accuracy_score(y_labels, y_pred)

    precision = precision_score(y_labels, y_pred, average='macro')
    f1 = f1_score(y_labels, y_pred, average='macro')
    recall = recall_score(y_labels, y_pred, average='macro') 

    write_log("\n--- HASIL METRIK LOOCV FINAL ---")
    write_log(f"Waktu total komputasi LOOCV: {total_hours:.2f} jam")
    write_log(f"Akurasi LOOCV: {acc * 100:.2f}%")
    write_log(f"Precision (macro): {precision:.4f}")
    write_log(f"F1-Score (macro): {f1:.4f}")
    write_log(f"Recall (macro): {recall:.4f}")

    write_log("\n--- LAPORAN KLASIFIKASI DETAIL ---")
    labels_az = [chr(ord('A') + i) for i in range(26)]
    report = classification_report(y_labels, y_pred, target_names=labels_az, digits=4)
    write_log(report)

    plot_confusion_matrix(y_labels, y_pred, svm_params['kernel'])

    return acc, precision, f1, y_pred 

def plot_confusion_matrix(y_true, y_pred, kernel_name):
    labels_az = [chr(ord('A') + i) for i in range(26)]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='pink',
                xticklabels=labels_az, yticklabels=labels_az,
                cbar=True, cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    title = f'Confusion Matrix (LOOCV + HOG + SVM {kernel_name.capitalize()})'
    plt.title(title, fontsize=16)
    plt.tight_layout()

    filename = f'confusion_matrix_hog_svm_{kernel_name}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    write_log(f"\n[VISUALISASI] Confusion Matrix disimpan ke: {filepath}")
    plt.close()

if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    write_log("=======================================================")
    write_log(f"Klasifikasi EMNIST Dimulai pada {time.ctime()}")
    write_log(f"Log Output: {LOG_FILE}")
    write_log("=======================================================")

    if not os.path.exists(FILE_PATH):
        write_log(f"ERROR: File {FILE_PATH} tidak ditemukan.")
    else:
        X, y = load_and_sample_data(FILE_PATH) 

        if RUN_TUNING_MODE:
            
            best_hog_params, best_svm_params = tune_parameters(X, y)
            
            HOG_PARAMS_FINAL.update(best_hog_params)
            SVM_PARAMS_FINAL.update(best_svm_params)
            
            write_log(f"\n[MENGAMBIL ULANG FITUR] Ekstraksi fitur dengan HOG params terbaik...")
            X_features_final = extract_hog_features(X, **HOG_PARAMS_FINAL)
        else:
            X_features_final = extract_hog_features(X, **HOG_PARAMS_FINAL)
            best_svm_params = SVM_PARAMS_FINAL


        if RUN_LOOCV_FINAL:
            acc, prec, f1, y_pred_final = evaluate_loocv_with_metrics(X_features_final, y, best_svm_params)
            write_log("\n--- DONE ---")