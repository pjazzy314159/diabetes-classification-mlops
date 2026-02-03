# Diabetes Prediction with MLflow

A machine learning project for diabetes prediction using ensemble methods, stacking, and hyperparameter optimization with Optuna, with full experiment tracking via MLflow.

## Overview

This project demonstrates a complete ML workflow for binary classification on imbalanced diabetes data. The pipeline includes data preprocessing, Stratified K-Fold cross-validation, SMOTE oversampling, hyperparameter tuning with Optuna, stacking ensemble, and experiment tracking.

## Key Optimizations

- **PR-AUC instead of ROC-AUC**: More suitable for imbalanced data evaluation
- **F2-Score optimization**: Prioritizes recall over precision (important for medical diagnosis)
- **Optuna hyperparameter tuning**: Automated tuning for base models and meta-learner
- **Custom threshold tuning**: Optimized classification threshold instead of default 0.5
- **Stratified K-Fold**: Maintains class distribution across folds

## Data

The dataset combines two sources with binary diabetes labels:
- **Total samples**: 324,372
- **Class distribution**: 76.8% non-diabetic, 23.2% diabetic
- **Features**: 21 features
- **Imbalance handling**: SMOTE oversampling on training set

## Workflow

1. Data preprocessing and concatenation of multiple data sources
2. Stratified K-Fold cross-validation (5 folds)
3. SMOTE applied on each fold's training set
4. **Optuna hyperparameter tuning** for base models (30 trials each):
   - XGBoost (eval_metric: PR-AUC)
   - LightGBM
   - CatBoost (eval_metric: PR-AUC)
5. Out-of-Fold (OOF) predictions collection
6. **Optuna tuning for meta-learner** (50 trials) with threshold optimization
7. Stacking with Logistic Regression as meta-learner
8. Threshold comparison and final evaluation

## MLflow Integration

All experiments are tracked using MLflow with SQLite backend:

- Experiment: `Diabetes Prediction - Stacking PR-AUC`

Logged items per run:
- **Parameters**: n_folds, threshold, metric type, SMOTE status, all Optuna best params
- **Metrics**: PR-AUC, accuracy, recall, precision, F1-score, F2-score
- **Artifacts**:
  - Base models (XGBoost, LightGBM, CatBoost)
  - Meta-learner with signature
  - `data_full.csv` - Full dataset
  - `feature_names.csv` - Feature list
  - `meta_features.csv` - OOF predictions
  - `optuna_results.csv` - All tuning trials
  - `config.json` - Model configurations

## Project Structure

```
Diabetes-mlflow/
├── data/
│   ├── db1.csv
│   ├── db2.csv
│   ├── data_full.csv
│   ├── meta_features.csv
│   ├── feature_names.csv
│   ├── optuna_results.csv
│   └── config.json
├── notebooks/
│   ├── main.ipynb
│   └── optimize.ipynb
├── mlflow.db
├── mlruns/
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pjazzy314159/diabetes-mlflow
cd diabetes-mlflow
```

### 2. Create virtual environment

```bash
conda create -n diabetes-mlflow python=3.10
conda activate diabetes-mlflow
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/optimize.ipynb
```

### 5. View MLflow UI

After running the notebook, start MLflow UI with SQLite backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open browser at `http://localhost:5000` to view experiments.

## Results

### Final Stacking Performance (Threshold = 0.201)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7355 |
| Precision | 0.4600 |
| Recall | 0.8004 |
| F1-Score | 0.5843 |
| F2-Score | 0.6972 |
| PR-AUC | 0.5858 |

### Base Models Average Metrics (5-Fold CV)

| Model | PR-AUC | Recall | Precision |
|-------|--------|--------|-----------|
| XGBoost | 0.5732 | 0.8926 | 0.3955 |
| LightGBM | 0.5654 | 0.8906 | 0.3890 |
| CatBoost | 0.5658 | 0.9081 | 0.3778 |

### Threshold Analysis

The optimized threshold (0.201) provides the best F2-score, balancing recall and precision for medical diagnosis where missing diabetic cases is more costly than false positives.