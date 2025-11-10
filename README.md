# EMNIST Handwritten Character Classification

Zaky Askar Sonia - 4212301088

This repository contains the implementation and evaluation results of a handwritten character classification system using the EMNIST dataset. The classification is performed using HOG feature extraction and SVM models, evaluated with the Leave-One-Out Cross-Validation (LOOCV) method.

## Data Processing
- **Dataset**: EMNIST Letters (`emnist-letters-train.csv`).
- **Sampling**: Balanced sampling of 500 samples per class.
- **Feature Extraction**: Histogram of Oriented Gradients (HOG) with parameters:
  - Orientations: 10
  - Pixels per cell (PPC): (8, 8)
  - Cells per block (CPB): (3, 3)

## LOOCV Evaluation Method
- **Model**: Support Vector Machine (SVM) with the following parameters:
  - Kernel: RBF
  - C: 10
  - Gamma: Scale
- **Evaluation**: Each sample is used as a test case while the rest are used for training.

## Evaluation Metrics
- **Accuracy**: 87.78%
- **Precision (macro)**: 0.8783
- **F1-Score (macro)**: 0.8779
- **Recall (macro)**: 0.8778

## Confusion Matrix
The confusion matrix visualizes the classification performance for each class. It is saved as `confusion_matrix_hog_svm_rbf.png` in the `Results1` directory.

## How to Run
1. Ensure the dataset file (`emnist-letters-train.csv`) is in the `archive` directory.
2. Run the `main.py` script to perform parameter tuning and LOOCV evaluation.