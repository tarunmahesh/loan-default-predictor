"""
Theorem-Style Credit Risk Pipeline
==================================
End-to-end underwriting model pipeline using LendingClub data.

Setup
-----
1. Download the dataset from Kaggle:
       kaggle datasets download -d wordsforthewise/lending-club
   or manually from:
       https://www.kaggle.com/datasets/wordsforthewise/lending-club

2. Unzip it:
       unzip lending-club.zip

3. Set DATA_FILE in the CONFIG block below to the path of
   accepted_2007_to_2018Q4.csv on your machine.

4. Run:
       python credit_risk_pipeline.py
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for VS Code terminals
                        # change to "TkAgg" or remove this line if you want
                        # a pop-up window instead of saving to file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG  ← all magic numbers live here
# ─────────────────────────────────────────────

CFG = dict(
    # ── data ────────────────────────────────────────────────────────────────
    # Path to the accepted loans CSV.  Edit this before running.
    data_file = Path("accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"),

    # ── model ────────────────────────────────────────────────────────────────
    n_estimators          = 2000,   # max trees; early stopping cuts this down
    learning_rate         = 0.03,
    num_leaves            = 63,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    early_stopping_rounds = 50,     # stop if val AUC stagnates for 50 rounds
    val_fraction          = 0.10,   # share of train set used for early-stopping

    # ── split ────────────────────────────────────────────────────────────────
    oot_date_percentile = 0.80,     # loans after this issue_d date → OOT test
    random_state        = 42,

    # ── profit curve scenarios  {label: (avg_loan, avg_profit, recovery_rate)}
    profit_scenarios = {
        "Base":        (15_000, 3_500, 0.40),
        "Optimistic":  (15_000, 4_500, 0.50),
        "Pessimistic": (15_000, 2_500, 0.30),
    },

    # ── explainability ───────────────────────────────────────────────────────
    shap_sample = 3_000,

    # ── cohort bins ──────────────────────────────────────────────────────────
    cohort_bins   = [0, 0.10, 0.20, 0.35, 1.0],
    cohort_labels = ["Low", "Medium", "High", "Very High"],

    # ── output ───────────────────────────────────────────────────────────────
    out_dir = Path("outputs"),
    dpi     = 150,
)

# ─────────────────────────────────────────────
# Feature lists
# ─────────────────────────────────────────────

# Categorical columns — kept as pandas.Categorical, handled natively by LightGBM
CAT_FEATURES = ["home_ownership", "purpose"]

# Numeric columns that exist directly in the CSV
RAW_NUM_FEATURES = [
    "loan_amnt", "term", "int_rate", "annual_inc", "dti",
    "fico_range_low", "emp_length", "open_acc", "pub_rec",
    "revol_util", "delinq_2yrs",
]

# Full numeric list (raw + engineered ratios)
NUM_FEATURES = RAW_NUM_FEATURES + ["loan_to_income", "credit_stress"]

# Everything the model sees
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw LendingClub columns and add engineered ratios."""
    # Select only raw source columns (engineered ones don't exist yet)
    f = frame[RAW_NUM_FEATURES + CAT_FEATURES + ["Default"]].copy()

    # Clean term and emp_length — extract the leading integer
    f["term"] = pd.to_numeric(
        f["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    f["emp_length"] = pd.to_numeric(
        f["emp_length"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(0)

    # Strip % signs
    for col in ["int_rate", "revol_util"]:
        f[col] = pd.to_numeric(
            f[col].astype(str).str.replace("%", "", regex=False), errors="coerce"
        )

    # Coerce remaining numeric columns
    for col in RAW_NUM_FEATURES:
        f[col] = pd.to_numeric(f[col], errors="coerce")

    # Engineered ratios (computed after cleaning so values are valid)
    f["loan_to_income"] = f["loan_amnt"] / f["annual_inc"].clip(lower=1)
    f["credit_stress"]  = f["dti"] * f["revol_util"]

    # Native categoricals — LightGBM handles these as unordered labels
    for col in CAT_FEATURES:
        f[col] = f[col].astype("category")

    return f.dropna(subset=NUM_FEATURES)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main() -> None:
    CFG["out_dir"].mkdir(exist_ok=True)

    print("=" * 60)
    print("THEOREM-STYLE CREDIT RISK PIPELINE")
    print("=" * 60)

    # ── Load & filter ────────────────────────────────────────────────────────

    data_file = CFG["data_file"]
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_file}'.\n"
            "Download it with:\n"
            "  kaggle datasets download -d wordsforthewise/lending-club\n"
            "  unzip lending-club.zip\n"
            "Then update CFG['data_file'] to the correct path."
        )

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file, low_memory=False)
    print(f"Rows loaded: {len(df):,}")

    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["Default"] = (df["loan_status"] == "Charged Off").astype(int)
    print(f"Default rate: {df['Default'].mean():.2%}")

    # ── Out-of-time split on issue_d ─────────────────────────────────────────

    print("\nBuilding out-of-time split...")
    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    cutoff_date = df["issue_d_parsed"].quantile(CFG["oot_date_percentile"])
    print(
        f"OOT cutoff ({CFG['oot_date_percentile']:.0%} percentile): "
        f"{cutoff_date.strftime('%b-%Y')}"
    )

    oot_mask     = df["issue_d_parsed"] >= cutoff_date
    df_train_raw = df[~oot_mask].copy()
    df_oot_raw   = df[ oot_mask].copy()
    print(f"Train (pre-cutoff): {len(df_train_raw):,}  |  OOT test: {len(df_oot_raw):,}")

    # ── Feature engineering ──────────────────────────────────────────────────

    print("\nEngineering features...")
    df_train_eng = engineer(df_train_raw)
    df_oot_eng   = engineer(df_oot_raw)

    # Align OOT category levels to training levels
    for col in CAT_FEATURES:
        df_oot_eng[col] = pd.Categorical(
            df_oot_eng[col], categories=df_train_eng[col].cat.categories
        )

    print(f"Train rows after cleaning : {len(df_train_eng):,}")
    print(f"OOT rows after cleaning   : {len(df_oot_eng):,}")

    # ── Train / val split for early stopping ─────────────────────────────────

    X_all = df_train_eng[ALL_FEATURES]
    y_all = df_train_eng["Default"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all,
        test_size    = CFG["val_fraction"],
        stratify     = y_all,
        random_state = CFG["random_state"],
    )

    X_oot = df_oot_eng[ALL_FEATURES]
    y_oot = df_oot_eng["Default"]

    print(f"\nTrain: {len(X_tr):,}  |  Val: {len(X_val):,}  |  OOT test: {len(X_oot):,}")

    # ── LightGBM with early stopping + native categoricals ───────────────────

    print("\nTraining LightGBM (early stopping on val AUC)...")

    base_model = lgb.LGBMClassifier(
        n_estimators     = CFG["n_estimators"],
        learning_rate    = CFG["learning_rate"],
        num_leaves       = CFG["num_leaves"],
        subsample        = CFG["subsample"],
        colsample_bytree = CFG["colsample_bytree"],
        class_weight     = "balanced",
        random_state     = CFG["random_state"],
    )
    base_model.fit(
        X_tr, y_tr,
        eval_set            = [(X_val, y_val)],
        eval_metric         = "auc",
        categorical_feature = CAT_FEATURES,
        callbacks           = [
            lgb.early_stopping(stopping_rounds=CFG["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = base_model.best_iteration_
    print(f"Best iteration (early stopping): {best_iter}")

    # ── Isotonic calibration ─────────────────────────────────────────────────

    print("Calibrating with isotonic regression...")

    final_base = lgb.LGBMClassifier(
        n_estimators        = best_iter,
        learning_rate       = CFG["learning_rate"],
        num_leaves          = CFG["num_leaves"],
        subsample           = CFG["subsample"],
        colsample_bytree    = CFG["colsample_bytree"],
        class_weight        = "balanced",
        random_state        = CFG["random_state"],
        categorical_feature = CAT_FEATURES,   # set in constructor, not fit()
    )

    model = CalibratedClassifierCV(final_base, method="isotonic", cv=3)
    model.fit(X_tr, y_tr)

    # ── Predictions + metrics (OOT) ──────────────────────────────────────────

    probs = model.predict_proba(X_oot)[:, 1]

    metrics = {
        "ROC-AUC (OOT)"          : roc_auc_score(y_oot, probs),
        "Average Precision (OOT)": average_precision_score(y_oot, probs),
        "Brier Score (OOT)"      : brier_score_loss(y_oot, probs),
    }

    print("\nModel metrics (out-of-time):")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v:.4f}")

    # ── Profit curve (multiple scenarios) ───────────────────────────────────

    print("\nComputing profit curves (3 scenarios)...")

    thresholds       = np.linspace(0.01, 0.99, 100)
    scenario_results = {}

    for label, (avg_loan, avg_profit, recovery) in CFG["profit_scenarios"].items():
        profits = np.array([
            (probs < t).sum() * 0   # placeholder; computed below
            for t in thresholds
        ], dtype=float)

        for i, t in enumerate(thresholds):
            approve    = probs < t
            good       = (approve & (y_oot.values == 0)).sum()
            bad        = (approve & (y_oot.values == 1)).sum()
            profits[i] = good * avg_profit - bad * avg_loan * (1 - recovery)

        best_idx = profits.argmax()
        scenario_results[label] = {
            "profits"   : profits,
            "threshold" : thresholds[best_idx],
            "max_profit": profits[best_idx],
        }
        print(
            f"  {label:<12} optimal threshold={thresholds[best_idx]:.3f}  "
            f"max profit=${profits[best_idx]/1e6:.2f}M"
        )

    best_threshold = scenario_results["Base"]["threshold"]

    # ── SHAP explainability ──────────────────────────────────────────────────

    print("\nComputing SHAP values...")

    inner_model  = model.calibrated_classifiers_[0].estimator
    explainer    = shap.TreeExplainer(inner_model)
    sample       = X_oot.sample(min(CFG["shap_sample"], len(X_oot)), random_state=CFG["random_state"])
    shap_values  = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── Cohort analysis ──────────────────────────────────────────────────────

    print("\nRunning cohort analysis...")

    cohort = pd.DataFrame({"prob": probs, "default": y_oot.values})
    cohort["tier"] = pd.cut(
        cohort["prob"],
        bins   = CFG["cohort_bins"],
        labels = CFG["cohort_labels"],
    )

    summary = cohort.groupby("tier", observed=True).agg(
        count        = ("default", "count"),
        default_rate = ("default", "mean"),
    )

    print("\nCohort analysis:")
    print(summary)

    # ── Dashboard ────────────────────────────────────────────────────────────

    print("\nGenerating dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Credit Risk Model Dashboard (Out-of-Time Evaluation)", fontsize=14, y=1.01)

    # Panel 1 — Score distribution
    axes[0, 0].hist(probs[y_oot == 0], bins=50, alpha=0.6, label="Fully Paid", color="#2196F3")
    axes[0, 0].hist(probs[y_oot == 1], bins=50, alpha=0.6, label="Default",    color="#F44336")
    axes[0, 0].axvline(
        best_threshold, color="black", linestyle="--", linewidth=1.2,
        label=f"Threshold = {best_threshold:.2f}",
    )
    axes[0, 0].set_title("Risk Score Distribution (OOT)")
    axes[0, 0].set_xlabel("Predicted Default Probability")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # Panel 2 — Profit curve sensitivity
    colors = {"Base": "#1976D2", "Optimistic": "#388E3C", "Pessimistic": "#D32F2F"}
    for label, res in scenario_results.items():
        axes[0, 1].plot(
            thresholds, res["profits"] / 1e6,
            label=f"{label} (t*={res['threshold']:.2f})",
            color=colors[label],
        )
        axes[0, 1].axvline(res["threshold"], linestyle=":", color=colors[label], linewidth=1)

    axes[0, 1].set_title("Profit Curve — Scenario Sensitivity")
    axes[0, 1].set_xlabel("Decision Threshold")
    axes[0, 1].set_ylabel("Portfolio Profit ($M)")
    axes[0, 1].legend(fontsize=9)

    # Panel 3 — Confusion matrix
    preds_opt = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_oot, preds_opt)
    im = axes[1, 0].imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            axes[1, 0].text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Pred Paid", "Pred Default"])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(["Actual Paid", "Actual Default"])
    axes[1, 0].set_title(f"Confusion Matrix @ t={best_threshold:.2f}")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Panel 4 — SHAP feature importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top_n    = 10

    axes[1, 1].barh(
        list(X_oot.columns[order][:top_n][::-1]),
        mean_abs[order][:top_n][::-1],
        color="#5C6BC0",
    )
    axes[1, 1].set_title("Top Risk Drivers (Mean |SHAP|)")
    axes[1, 1].set_xlabel("Mean |SHAP value|")

    plt.tight_layout()
    dashboard_path = CFG["out_dir"] / "dashboard.png"
    plt.savefig(dashboard_path, dpi=CFG["dpi"], bbox_inches="tight")
    print(f"Dashboard saved → {dashboard_path}")
    plt.close()

    # ── SHAP summary plot ─────────────────────────────────────────────────────

    print("\nSaving SHAP summary...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    shap_path = CFG["out_dir"] / "shap_summary.png"
    plt.savefig(shap_path, dpi=CFG["dpi"])
    print(f"SHAP summary saved → {shap_path}")
    plt.close()

    # ── Save outputs ──────────────────────────────────────────────────────────

    summary.to_csv(CFG["out_dir"] / "cohort_analysis.csv")

    with open(CFG["out_dir"] / "metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    scenario_export = {
        label: {
            "optimal_threshold": float(res["threshold"]),
            "max_profit_usd":    float(res["max_profit"]),
        }
        for label, res in scenario_results.items()
    }
    with open(CFG["out_dir"] / "profit_scenarios.json", "w") as f:
        json.dump(scenario_export, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for k, v in metrics.items():
        print(f"  {k:<30}: {v:.4f}")

    print(f"\n  Best iteration (early stop) : {best_iter}")
    print(f"  Optimal threshold (Base)    : {best_threshold:.3f}")
    print(f"\n  Dashboard                   : {dashboard_path}")
    print(f"  SHAP summary                : {shap_path}")
    print(f"  Cohort CSV                  : {CFG['out_dir']}/cohort_analysis.csv")
    print(f"  Profit scenarios JSON       : {CFG['out_dir']}/profit_scenarios.json")
    print("\nAll outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
