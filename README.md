# Machine-Learning-Based-Diabetes-Prediction-with-Loss-Optimization-using-Boosted-Coati-Optimization
Reproducible ML pipeline for Pima diabetes prediction with ECO-style XGBoost log-loss optimization, evaluation metrics, and plots
# Diabetes Prediction with ECO-Tuned XGBoost (Log-Loss Optimization)

This repository contains a **reproducible machine learning pipeline** for **diabetes prediction** using the **Pima Indians Diabetes dataset**.  
It compares baseline models (Logistic Regression, Decision Tree, Random Forest) against **XGBoost**, and proposes an **ECO-style hyperparameter tuning** approach that optimizes **cross-validated log loss** to improve **probability quality** and screening suitability.

## Key Features
- ✅ Leakage-free preprocessing using **sklearn Pipelines**
- ✅ **Zero-as-missing** handling for physiologically invalid zeros (Glucose, BP, SkinThickness, Insulin, BMI)
- ✅ Robust **median imputation + missing indicators + MinMax scaling** (fit on train only)
- ✅ Interaction feature engineering: `Glucose_BMI`, `Age_Glucose`, `Insulin_BMI`
- ✅ ECO-style tuning for XGBoost using **CV LogLoss**
- ✅ Full evaluation artifacts: confusion matrices, ROC curves, metrics tables, logloss tables, CV summaries, feature importances
- ✅ Excel summary output: `results_summary.xlsx`

---

## Project Structure
.
├── diabetes_eco_pipeline_updated.py
├── outputs/ (auto-generated after running)
│ ├── missing_value_heatmap.png
│ ├── correlation_heatmap.png
│ ├── outcome_correlation.png
│ ├── pairplot.png
│ ├── CM_.png
│ ├── ROC_.png
│ ├── Feature_Importance_*.png
│ ├── final_metrics_table.csv
│ ├── logloss_table.csv
│ ├── train_test_accuracy.csv
│ ├── cv10_summary.csv
│ ├── mean_median_std_summary.csv
│ ├── eco_search_history.csv
│ ├── threshold_search.csv
│ ├── results_summary.xlsx
│ ├── best_model_threshold_tuned.joblib
│ └── xgb_proposed_eco_pipeline.joblib
└── data/
└── diabetes.csv (you provide)


---

## Dataset
- Dataset: **Pima Indians Diabetes Database**
- Source: Kaggle / UCI ML Repository mirror
- File expected: `diabetes.csv` with columns:
  - `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

---

## Installation

### 1) Create environment (recommended)

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
2) Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost openpyxl joblib
(Optional) requirements.txt:

numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
joblib
How to Run
Basic run
python diabetes_eco_pipeline_updated.py --data "D:\DiabetesML\diabetes.csv"
Specify output folder
python diabetes_eco_pipeline_updated.py --data "D:\DiabetesML\diabetes.csv" --out outputs
Control ECO tuning iterations + CV folds
python diabetes_eco_pipeline_updated.py --data "D:\DiabetesML\diabetes.csv" --eco_iters 35 --eco_cv 5
Threshold tuning objective (screening-focused)
f2 (recommended): emphasizes recall

f1: balanced

recall: maximize recall

python diabetes_eco_pipeline_updated.py --data "D:\DiabetesML\diabetes.csv" --thr_metric f2
Outputs (What to Use in Your Paper)
Tables (CSV/Excel)
outputs/final_metrics_table.csv → Accuracy, Precision, Recall, F1, LogLoss, ROC_AUC + TN/FP/FN/TP

outputs/logloss_table.csv → Log loss comparison (sorted)

outputs/train_test_accuracy.csv → Train vs test accuracy/recall (overfitting check)

outputs/cv10_summary.csv → 10-fold CV mean/median/std (stability)

outputs/eco_search_history.csv → ECO candidate hyperparameters + mean CV logloss (best row = tuned params)

outputs/results_summary.xlsx → All key tables in one Excel file

Figures (PNG)
Missingness: missing_value_heatmap.png

Correlations: correlation_heatmap.png, outcome_correlation.png

Pairplot: pairplot.png

Confusion matrices: CM_*.png

ROC: ROC_*.png, ROC_All_Models.png

Comparisons: 05_model_comparison_bar.png, 06_logloss_comparison_bar.png

Feature importance: Feature_Importance_*.png

Leakage-Free Notes
Zeros are converted to missing without using Outcome.

Imputation/scaling are inside Pipelines, so they are fit on training folds only during CV.

ECO tuning uses cross-validated LogLoss on training only (no test leakage).


Author
Sameer Sethi
B.Tech CSE — Sikkim Manipal Institute of Technology
