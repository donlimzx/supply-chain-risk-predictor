"""
Phase 2 — Supply Chain Late-Delivery Risk Model
================================================
Trains a Logistic Regression baseline and an XGBoost classifier on the
DataCo Supply Chain dataset, evaluates both, and persists the best model.

Usage:
    python src/train_model.py
"""

import os
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "DataCoSupplyChainDataset.csv"
MODEL_PATH = ROOT / "src" / "model.pkl"
FIGURES_DIR = ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD ────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding="latin-1",
                 parse_dates=["order date (DateOrders)"])
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────

df["order_month"] = df["order date (DateOrders)"].dt.month
df["order_dow"]   = df["order date (DateOrders)"].dt.dayofweek   # 0=Mon

# Safe numeric features (all known at order-placement time)
NUM_FEATURES = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
    "order_month",
    "order_dow",
]

# Safe categorical features
CAT_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment",
    "Department Name",
    "Category Name",
    "Type",            # payment type (DEBIT / TRANSFER / etc.)
]

TARGET = "Late_delivery_risk"

# Drop any rows with nulls in selected columns (negligible: <10 rows)
keep_cols = NUM_FEATURES + CAT_FEATURES + [TARGET]
df_model = df[keep_cols].dropna()
print(f"  {df_model.shape[0]:,} rows after dropping nulls in feature set")

X = df_model[NUM_FEATURES + CAT_FEATURES]
y = df_model[TARGET]

print(f"\nClass distribution:\n{y.value_counts().to_string()}")
print(f"  Late rate: {y.mean():.1%}")

# ── 3. TRAIN / TEST SPLIT ──────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. PREPROCESSOR ────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
])

# ── 5. LOGISTIC REGRESSION BASELINE ───────────────────────────────────────

print("\n" + "="*60)
print("LOGISTIC REGRESSION  (baseline)")
print("="*60)

lr_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
])

lr_pipe.fit(X_train, y_train)
y_pred_lr  = lr_pipe.predict(X_test)
y_prob_lr  = lr_pipe.predict_proba(X_test)[:, 1]

lr_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr),
    "Recall":    recall_score(y_test, y_pred_lr),
    "F1":        f1_score(y_test, y_pred_lr),
    "AUC-ROC":   roc_auc_score(y_test, y_prob_lr),
}

for k, v in lr_metrics.items():
    print(f"  {k:<12} {v:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred_lr, target_names=["On time/early", "Late"]))

# ── 6. XGBOOST ─────────────────────────────────────────────────────────────

print("="*60)
print("XGBOOST")
print("="*60)

# scale_pos_weight compensates for mild class imbalance
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

xgb_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )),
])

xgb_pipe.fit(X_train, y_train)
y_pred_xgb = xgb_pipe.predict(X_test)
y_prob_xgb = xgb_pipe.predict_proba(X_test)[:, 1]

xgb_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_xgb),
    "Precision": precision_score(y_test, y_pred_xgb),
    "Recall":    recall_score(y_test, y_pred_xgb),
    "F1":        f1_score(y_test, y_pred_xgb),
    "AUC-ROC":   roc_auc_score(y_test, y_prob_xgb),
}

for k, v in xgb_metrics.items():
    print(f"  {k:<12} {v:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred_xgb, target_names=["On time/early", "Late"]))

# ── 7. COMPARISON TABLE ────────────────────────────────────────────────────

print("="*60)
print("MODEL COMPARISON")
print("="*60)
comp = pd.DataFrame({"Logistic Regression": lr_metrics, "XGBoost": xgb_metrics}).T
print(comp.to_string(float_format="{:.4f}".format))

best_model_name = "XGBoost" if xgb_metrics["AUC-ROC"] >= lr_metrics["AUC-ROC"] else "Logistic Regression"
best_pipe       = xgb_pipe   if best_model_name == "XGBoost" else lr_pipe
best_metrics    = xgb_metrics if best_model_name == "XGBoost" else lr_metrics
print(f"\nBest model: {best_model_name}  (AUC-ROC={best_metrics['AUC-ROC']:.4f})")

# ── 8. FIGURES ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ROC curves
for name, y_prob, color in [
    ("Logistic Regression", y_prob_lr, "steelblue"),
    ("XGBoost",             y_prob_xgb, "tomato"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", color=color, linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Confusion matrices side-by-side
for ax, name, y_pred in [
    (axes[1], "Logistic Regression", y_pred_lr),
    (axes[2], "XGBoost",             y_pred_xgb),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["On time", "Late"],
                yticklabels=["On time", "Late"])
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "phase2_roc_cm.png", dpi=110)
print(f"\nFigure saved → {FIGURES_DIR / 'phase2_roc_cm.png'}")

# XGBoost feature importances
if best_model_name == "XGBoost":
    ohe_cats = (xgb_pipe.named_steps["pre"]
                .named_transformers_["cat"]
                .get_feature_names_out(CAT_FEATURES))
    feat_names = np.array(NUM_FEATURES + list(ohe_cats))
    importances = xgb_pipe.named_steps["clf"].feature_importances_

    top_n = 20
    idx = np.argsort(importances)[-top_n:]
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    ax2.barh(feat_names[idx], importances[idx], color="tomato", edgecolor="white")
    ax2.set_title(f"XGBoost — Top {top_n} Feature Importances")
    ax2.set_xlabel("Importance (gain)")
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "phase2_feature_importance.png", dpi=110)
    print(f"Figure saved → {FIGURES_DIR / 'phase2_feature_importance.png'}")

# ── 9. SAVE MODEL ──────────────────────────────────────────────────────────

payload = {
    "model":        best_pipe,
    "model_name":   best_model_name,
    "metrics":      best_metrics,
    "num_features": NUM_FEATURES,
    "cat_features": CAT_FEATURES,
    "target":       TARGET,
}

joblib.dump(payload, MODEL_PATH)

print(f"\nModel saved → {MODEL_PATH}")
print("\nDone.")
