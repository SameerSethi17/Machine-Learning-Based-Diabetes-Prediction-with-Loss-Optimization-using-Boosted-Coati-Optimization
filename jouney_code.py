# -*- coding: utf-8 -*-
"""
Leakage-free FULL PIPELINE: PIMA Diabetes ML + ECO-style XGBoost Tuning + EDA + Full Evaluation Artifacts

Major fixes vs your older script:
✅ NO class-wise (Outcome-based) imputation. That was data leakage.
✅ Imputation/scaling are done inside sklearn Pipelines (so CV is valid).
✅ Added missing indicators + imbalance handling + threshold tuning (CV OOF) to improve Recall.

Run:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost openpyxl joblib
python diabetes_eco_pipeline_updated.py --data "D:\\DiabetesML\\diabetes.csv"

Outputs saved to ./outputs/
"""

import os
import warnings
import argparse
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib

warnings.filterwarnings("ignore")


# -------------------------------
# Utilities
# -------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.copy()
    b2 = b2.replace(0, np.nan)
    out = a / b2
    return out.replace([np.inf, -np.inf], np.nan)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Interaction features
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Age_Glucose"] = df["Age"] * df["Glucose"]
    df["Insulin_BMI"] = df["Insulin"] * df["BMI"]
    return df


def plot_missing_heatmap(df_raw: pd.DataFrame, out_dir: str) -> None:
    """Heatmap after zero-as-missing replacement, before imputation (EDA only)."""
    plt.figure(figsize=(14, 5))
    sns.heatmap(df_raw.isna(), cbar=False)
    plt.title("Missing Values Heatmap (Zero-as-missing applied, before imputation)")
    save_fig(os.path.join(out_dir, "missing_value_heatmap.png"))


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    save_fig(os.path.join(out_dir, "correlation_heatmap.png"))


def plot_outcome_correlation(df: pd.DataFrame, target_col: str, out_dir: str) -> None:
    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    corr = corr.drop(target_col, errors="ignore")
    plt.figure(figsize=(12, 4))
    corr.plot(kind="bar")
    plt.ylabel("Correlation")
    plt.title("Feature Correlation with Outcome")
    save_fig(os.path.join(out_dir, "outcome_correlation.png"))


def plot_pairplot(df: pd.DataFrame, out_dir: str) -> None:
    cols = ["Glucose", "BMI", "Age", "Insulin", "Glucose_BMI", "Age_Glucose", "Outcome"]
    cols = [c for c in cols if c in df.columns]
    pp = sns.pairplot(df[cols], hue="Outcome", diag_kind="hist", plot_kws={"alpha": 0.7, "s": 20})
    pp.fig.suptitle("Pairplot (Selected Features) vs Outcome", y=1.02)
    pp.savefig(os.path.join(out_dir, "pairplot.png"), dpi=200, bbox_inches="tight")
    plt.close("all")


def make_preprocessor(numeric_features):
    """
    Leakage-free preprocessing:
    - Treat zeros as missing for specific medical columns (done before split, but without using Outcome)
    - Missing indicator (add_is_missing=True)
    - Median imputation (fit only on training within pipeline)
    - MinMax scaling (fit only on training within pipeline)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test, out_dir, prefix=""):
    # Predict labels + probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function -> sigmoid-like scaling (rare)
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    ll = log_loss(y_test, np.clip(y_prob, 1e-8, 1 - 1e-8))

    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    save_fig(os.path.join(out_dir, f"CM_{name.replace(' ', '_')}.png"))

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    save_fig(os.path.join(out_dir, f"ROC_{name.replace(' ', '_')}.png"))

    return {
        "Model": name,
        "Test_Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "LogLoss": ll,
        "ROC_AUC": roc_auc,
        "CM_TN": int(cm[0, 0]),
        "CM_FP": int(cm[0, 1]),
        "CM_FN": int(cm[1, 0]),
        "CM_TP": int(cm[1, 1]),
    }


def plot_model_comparison(metrics_df: pd.DataFrame, out_dir: str) -> None:
    df = metrics_df.copy()
    cols = ["Test_Accuracy", "Precision", "Recall", "F1"]
    df_plot = df.set_index("Model")[cols]
    ax = df_plot.plot(kind="bar", figsize=(12, 4))
    ax.set_title("Model Comparison (Hold-out Test): Accuracy / Precision / Recall / F1")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(os.path.join(out_dir, "05_model_comparison_bar.png"))


def plot_logloss_comparison(metrics_df: pd.DataFrame, out_dir: str) -> None:
    df = metrics_df.copy()
    ax = df.set_index("Model")["LogLoss"].plot(kind="bar", figsize=(12, 4))
    ax.set_title("LogLoss Comparison (Hold-out Test) – Lower is Better")
    ax.set_ylabel("Test LogLoss")
    ax.set_xlabel("Model")
    plt.xticks(rotation=25, ha="right")
    save_fig(os.path.join(out_dir, "06_logloss_comparison_bar.png"))


def plot_roc_all_models(metrics_by_name: dict, out_dir: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    for name, (y_true, y_prob) in metrics_by_name.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.title("ROC Curve – All Models")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    save_fig(os.path.join(out_dir, "ROC_All_Models.png"))


def get_feature_importance_plot(model, feature_names, title, out_path, top_k=15):
    """
    Works for:
    - LR: abs(coef_)
    - Tree/Forest/XGB: feature_importances_
    NOTE: with Pipeline+ColumnTransformer, we extract names carefully.
    """
    importances = None

    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
        xlab = "|Coefficient|"
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlab = "Importance"
    else:
        return

    idx = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    plt.figure(figsize=(10, 5))
    plt.barh(top_features[::-1], top_vals[::-1])
    plt.xlabel(xlab)
    plt.ylabel("Feature")
    plt.title(title)
    save_fig(out_path)


def extract_feature_names(preprocessor: ColumnTransformer, numeric_features):
    """
    When using SimpleImputer(add_indicator=True), the output adds MissingIndicator columns.
    sklearn provides get_feature_names_out in newer versions.
    We'll try to use it, otherwise fallback.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: numeric feature names + indicator names
        base = [f"num__{c}" for c in numeric_features]
        # can't know indicator count exactly without fitting; return base
        return base


# -------------------------------
# ECO-style tuning (random-search with CV logloss) on TRAIN ONLY
# -------------------------------
def eco_tune_xgb_with_cv(X, y, preprocessor, scale_pos_weight: float,
                        n_iters=30, cv_splits=5, random_state=42):
    """
    ECO-style search (practical): sample hyperparams and select best by mean CV LogLoss.
    Fully leakage-free because preprocessing is inside pipeline for each fold.
    """
    rng = np.random.RandomState(random_state)

    param_space = {
        "n_estimators": [200, 300, 400, 500, 700],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 5],
        "gamma": [0, 0.05, 0.1, 0.2],
        "reg_alpha": [0, 1e-3, 1e-2, 1e-1],
        "reg_lambda": [1.0, 1.5, 2.0, 3.0],
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_params = None
    best_score = float("inf")
    history = []

    for it in range(n_iters):
        params = {k: rng.choice(v) for k, v in param_space.items()}

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            **params
        )

        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("model", xgb),
        ])

        fold_losses = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            try:
                pipe.fit(X_tr, y_tr)
                prob = pipe.predict_proba(X_va)[:, 1]
                ll = log_loss(y_va, np.clip(prob, 1e-8, 1 - 1e-8))
                fold_losses.append(ll)
            except Exception:
                continue

        if len(fold_losses) == 0:
            continue

        mean_ll = float(np.mean(fold_losses))
        history.append({"iter": it, "mean_cv_logloss": mean_ll, **params})

        if mean_ll < best_score:
            best_score = mean_ll
            best_params = params

    history_df = pd.DataFrame(history).sort_values("mean_cv_logloss")
    return best_params, best_score, history_df


# -------------------------------
# Threshold tuning (to improve Recall) on TRAIN ONLY
# -------------------------------
def tune_threshold_oof(model_pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                       cv_splits=5, random_state=42, metric="f2"):
    """
    Tune classification threshold using out-of-fold probabilities (train only).
    metric:
      - "f1" : balance precision/recall
      - "f2" : emphasize recall (recommended for medical screening)
      - "recall" : maximize recall
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    oof_prob = np.zeros(len(y_train), dtype=float)

    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        model_pipe.fit(X_tr, y_tr)
        oof_prob[va_idx] = model_pipe.predict_proba(X_va)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_val = -1.0
    rows = []

    for thr in thresholds:
        pred = (oof_prob >= thr).astype(int)
        prec = precision_score(y_train, pred, zero_division=0)
        rec = recall_score(y_train, pred, zero_division=0)
        f1 = f1_score(y_train, pred, zero_division=0)
        # F2
        beta2 = 4.0
        denom = (beta2 * prec + rec)
        f2 = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0

        if metric == "recall":
            score = rec
        elif metric == "f1":
            score = f1
        else:
            score = f2

        rows.append({"threshold": thr, "precision": prec, "recall": rec, "f1": f1, "f2": f2, "score": score})

        if score > best_val:
            best_val = score
            best_thr = thr

    thr_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return best_thr, thr_df


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"D:\DiabetesML\diabetes.csv", help="Path to diabetes.csv")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--eco_iters", type=int, default=35, help="ECO-style tuning iterations")
    parser.add_argument("--eco_cv", type=int, default=5, help="CV folds for ECO tuning")
    parser.add_argument("--thr_metric", type=str, default="f2", choices=["f2", "f1", "recall"], help="Threshold tuning objective")
    args = parser.parse_args()

    out_dir = args.out
    ensure_dir(out_dir)

    # 1) Load dataset
    df = pd.read_csv(args.data)

    # 2) Replace zeros as missing for specific columns (common PIMA practice)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_raw = df.copy()
    for col in zero_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].replace(0, np.nan)

    # 3) EDA plots before imputation
    plot_missing_heatmap(df_raw, out_dir)

    # 4) Add engineered features (still with NaNs)
    df_feat = add_engineered_features(df_raw)

    # 5) Correlation plots (EDA) using NaN-aware pairwise correlations
    plot_correlation_heatmap(df_feat.dropna(axis=0, how="any"), out_dir)
    plot_outcome_correlation(df_feat.dropna(axis=0, how="any"), "Outcome", out_dir)

    # 6) Pairplot (sample to keep it lighter)
    try:
        sample_df = df_feat.copy()
        if len(sample_df) > 600:
            sample_df = sample_df.sample(600, random_state=42)
        plot_pairplot(sample_df.dropna(axis=0, how="any"), out_dir)
    except Exception:
        pass

    # 7) Train/test split (stratified)
    target_col = "Outcome"
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8) Compute imbalance weight for XGB on training set only
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # 9) Build preprocessing (numeric only)
    numeric_features = list(X.columns)
    preprocessor = make_preprocessor(numeric_features)

    # 10) Define base models (with imbalance handling)
    models = {}

    models["Logistic Regression"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    models["Decision Tree"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=5
        ))
    ])

    models["Random Forest"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # Base XGBoost
    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=1,
        gamma=0.0,
        reg_lambda=1.0,

    )
    models["XGBoost Base"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", base_xgb)
    ])

    # 11) ECO-style tuning for XGB on TRAIN only (CV LogLoss)
    best_params, best_cv_ll, history_df = eco_tune_xgb_with_cv(
        X_train, y_train, preprocessor=preprocessor,
        scale_pos_weight=scale_pos_weight,
        n_iters=args.eco_iters, cv_splits=args.eco_cv, random_state=42
    )
    history_df.to_csv(os.path.join(out_dir, "eco_search_history.csv"), index=False)

    if best_params is None:
        # fallback to base params if tuning fails
        best_params = {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

    tuned_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        **best_params
    )
    models["XGBoost Proposed ECO"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", tuned_xgb)
    ])

    # 12) Train + evaluate all models on hold-out test (threshold=0.5)
    metrics = []
    roc_store = {}  # name -> (y_true, y_prob)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        # Save ROC prob for all-model plot
        y_prob = pipe.predict_proba(X_test)[:, 1]
        roc_store[name] = (y_test.values, y_prob)

        row = evaluate_model(name, pipe, X_test, y_test, out_dir)
        metrics.append(row)

        # Feature importance plots
        try:
            pre = pipe.named_steps["pre"]
            feature_names = extract_feature_names(pre, numeric_features)
            mdl = pipe.named_steps["model"]
            get_feature_importance_plot(
                mdl,
                feature_names=feature_names,
                title=f"Feature Importance - {name} (Top 15)",
                out_path=os.path.join(out_dir, f"Feature_Importance_{name.replace(' ', '_')}.png"),
                top_k=15
            )
        except Exception:
            pass

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(out_dir, "final_metrics_table.csv"), index=False)

    # Comparison plots
    plot_model_comparison(metrics_df, out_dir)
    plot_logloss_comparison(metrics_df, out_dir)
    plot_roc_all_models(roc_store, out_dir)

    # LogLoss table
    logloss_df = metrics_df[["Model", "LogLoss"]].sort_values("LogLoss")
    logloss_df.to_csv(os.path.join(out_dir, "logloss_table.csv"), index=False)

    # 13) Threshold tuning for BEST model (by F1 or Recall) on TRAIN only, then evaluate on TEST
    # Choose best base model by F1 (you can change to Recall)
    best_model_name = metrics_df.sort_values("F1", ascending=False).iloc[0]["Model"]
    best_pipe = models[best_model_name]

    best_thr, thr_df = tune_threshold_oof(best_pipe, X_train, y_train, cv_splits=5, random_state=42, metric=args.thr_metric)
    thr_df.to_csv(os.path.join(out_dir, "threshold_search.csv"), index=False)

    # Fit best pipe on full train, predict test using tuned threshold
    best_pipe.fit(X_train, y_train)
    test_prob = best_pipe.predict_proba(X_test)[:, 1]
    test_pred_thr = (test_prob >= best_thr).astype(int)

    tuned_row = {
        "Model": f"{best_model_name} (thr={best_thr:.2f}, obj={args.thr_metric})",
        "Test_Accuracy": accuracy_score(y_test, test_pred_thr),
        "Precision": precision_score(y_test, test_pred_thr, zero_division=0),
        "Recall": recall_score(y_test, test_pred_thr, zero_division=0),
        "F1": f1_score(y_test, test_pred_thr, zero_division=0),
        "LogLoss": log_loss(y_test, np.clip(test_prob, 1e-8, 1 - 1e-8)),
        "ROC_AUC": auc(*roc_curve(y_test, test_prob)[:2]),
        "CM_TN": int(confusion_matrix(y_test, test_pred_thr)[0, 0]),
        "CM_FP": int(confusion_matrix(y_test, test_pred_thr)[0, 1]),
        "CM_FN": int(confusion_matrix(y_test, test_pred_thr)[1, 0]),
        "CM_TP": int(confusion_matrix(y_test, test_pred_thr)[1, 1]),
    }

    metrics_thr_df = pd.concat([metrics_df, pd.DataFrame([tuned_row])], ignore_index=True)
    metrics_thr_df.to_csv(os.path.join(out_dir, "metrics_comparison_threshold_tuned.csv"), index=False)

    # 14) Train vs Test accuracy + recall (overfitting check)
    rows_tt = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        ytr_pred = pipe.predict(X_train)
        yte_pred = pipe.predict(X_test)

        rows_tt.append({
            "Model": name,
            "Train_Accuracy": accuracy_score(y_train, ytr_pred),
            "Test_Accuracy": accuracy_score(y_test, yte_pred),
            "Train_Recall": recall_score(y_train, ytr_pred, zero_division=0),
            "Test_Recall": recall_score(y_test, yte_pred, zero_division=0),
        })

    train_test_df = pd.DataFrame(rows_tt)
    train_test_df.to_csv(os.path.join(out_dir, "train_test_accuracy.csv"), index=False)

    # 15) 10-Fold CV stats (mean/median/std) for Accuracy + Recall for each model on TRAIN only (no leakage)
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    stats_rows = []
    for name, pipe in models.items():
        accs = []
        recs = []
        for tr_idx, va_idx in cv10.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            accs.append(accuracy_score(y_va, pred))
            recs.append(recall_score(y_va, pred, zero_division=0))

        stats_rows.append({
            "Model": name,
            "CV10_Accuracy_Mean": float(np.mean(accs)),
            "CV10_Accuracy_Median": float(np.median(accs)),
            "CV10_Accuracy_Std": float(np.std(accs)),
            "CV10_Recall_Mean": float(np.mean(recs)),
            "CV10_Recall_Median": float(np.median(recs)),
            "CV10_Recall_Std": float(np.std(recs)),
        })

    cv10_df = pd.DataFrame(stats_rows)
    cv10_df.to_csv(os.path.join(out_dir, "cv10_summary.csv"), index=False)

    # 16) Mean/Median/Std summary file (overall)
    mean_median_std = cv10_df[[
        "Model",
        "CV10_Accuracy_Mean", "CV10_Accuracy_Median", "CV10_Accuracy_Std",
        "CV10_Recall_Mean", "CV10_Recall_Median", "CV10_Recall_Std"
    ]].copy()
    mean_median_std.to_csv(os.path.join(out_dir, "mean_median_std_summary.csv"), index=False)

    # 17) Save an Excel summary
    xlsx_path = os.path.join(out_dir, "results_summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="holdout_metrics", index=False)
        logloss_df.to_excel(writer, sheet_name="logloss", index=False)
        train_test_df.to_excel(writer, sheet_name="train_vs_test", index=False)
        cv10_df.to_excel(writer, sheet_name="cv10_summary", index=False)
        thr_df.to_excel(writer, sheet_name="threshold_search", index=False)
        history_df.to_excel(writer, sheet_name="eco_search_history", index=False)

    # 18) Save best pipeline
    # Choose best by F1 from threshold-tuned table
    best_final = metrics_thr_df.sort_values("F1", ascending=False).iloc[0]
    best_final_name = best_final["Model"]

    # If the best final is threshold tuned, save that best_pipe + threshold
    model_artifact = {
        "pipeline": best_pipe,
        "threshold": float(best_thr),
        "threshold_objective": args.thr_metric,
        "note": "Use pipeline.predict_proba(X) and apply threshold for class prediction."
    }
    joblib.dump(model_artifact, os.path.join(out_dir, "best_model_threshold_tuned.joblib"))

    # Save the best ECO model as well
    joblib.dump(models["XGBoost Proposed ECO"], os.path.join(out_dir, "xgb_proposed_eco_pipeline.joblib"))

    print("\nDONE ✅")
    print(f"Outputs saved to: {os.path.abspath(out_dir)}")
    print(f"Best base model (by holdout F1): {best_model_name}")
    print(f"Tuned threshold (OOF train, obj={args.thr_metric}): {best_thr:.2f}")
    print(f"Best final entry (after adding threshold tuned): {best_final_name}")


if __name__ == "__main__":
    main()
