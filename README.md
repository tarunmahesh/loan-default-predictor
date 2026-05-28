# Credit Risk Detector

An end-to-end credit risk underwriting pipeline built on LendingClub loan data. It trains a gradient-boosted model to predict loan defaults, evaluates it on an out-of-time holdout set, and produces a full suite of business and explainability outputs.

---

## Features

- **Out-of-time (OOT) evaluation** — splits data chronologically so the model is always tested on loans issued after its training window, simulating real deployment conditions
- **LightGBM with early stopping** — trains up to 2,000 trees, halting when validation AUC stops improving
- **Isotonic probability calibration** — wraps the final model in a 3-fold `CalibratedClassifierCV` to produce reliable default probabilities
- **Multi-scenario profit curves** — sweeps decision thresholds under Base, Optimistic, and Pessimistic economic assumptions to find the profit-maximising approval cutoff
- **SHAP explainability** — generates feature importance and a summary beeswarm plot using TreeExplainer
- **Cohort analysis** — buckets borrowers into Low / Medium / High / Very High risk tiers and reports observed default rates per tier
- **Dashboard** — saves a 2×2 PNG with score distribution, profit curve sensitivity, confusion matrix, and top SHAP drivers

---

## Requirements

Python 3.8+ and the following packages:

```
lightgbm
shap
scikit-learn
pandas
numpy
matplotlib
```

Install all at once:

```bash
brew install lightgbm
pip install shap scikit-learn pandas numpy matplotlib
```

---

## Data Setup

The pipeline uses the public **LendingClub accepted loans** dataset from Kaggle.

1. Download the dataset:

   ```bash
   kaggle datasets download -d wordsforthewise/lending-club
   ```
   Or download manually from https://www.kaggle.com/datasets/wordsforthewise/lending-club

2. Unzip the archive:
   ```bash
   unzip lending-club.zip
   ```

3. Open `credit_risk_detector.py` and update the `data_file` path in the `CFG` block at the top of the file:
   ```python
   data_file = Path("path/to/accepted_2007_to_2018Q4.csv"),
   ```

Only rows with a `loan_status` of `"Fully Paid"` or `"Charged Off"` are used. All other statuses (current loans, in-grace-period, etc.) are filtered out.

---

## Usage

```bash
python credit_risk_detector.py
```

The script prints progress to stdout and writes all outputs to the `outputs/` directory (created automatically).

---

## Configuration

All tunable parameters are collected in the `CFG` dict near the top of the file — no magic numbers elsewhere.

| Key | Default | Description |
|-----|---------|-------------|
| `data_file` | *(path to CSV)* | Path to the LendingClub accepted loans CSV |
| `n_estimators` | `2000` | Maximum number of LightGBM trees |
| `learning_rate` | `0.03` | Gradient boosting step size |
| `num_leaves` | `63` | Max leaves per tree (controls model complexity) |
| `subsample` | `0.8` | Fraction of rows sampled per tree |
| `colsample_bytree` | `0.8` | Fraction of features sampled per tree |
| `early_stopping_rounds` | `50` | Stop training if val AUC doesn't improve for this many rounds |
| `val_fraction` | `0.10` | Share of training data held out for early stopping |
| `oot_date_percentile` | `0.80` | Loans issued after this date percentile become the OOT test set |
| `random_state` | `42` | Global random seed |
| `profit_scenarios` | Base / Optimistic / Pessimistic | Tuples of `(avg_loan, avg_profit, recovery_rate)` for profit curve simulation |
| `shap_sample` | `3000` | Number of OOT rows used to compute SHAP values |
| `cohort_bins` | `[0, 0.10, 0.20, 0.35, 1.0]` | Probability thresholds for risk tier bucketing |
| `out_dir` | `outputs/` | Directory for all saved outputs |

---

## Features Used

### Raw numeric features (sourced directly from the CSV)
`loan_amnt`, `term`, `int_rate`, `annual_inc`, `dti`, `fico_range_low`, `emp_length`, `open_acc`, `pub_rec`, `revol_util`, `delinq_2yrs`

### Engineered features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `loan_to_income` | `loan_amnt / annual_inc` | Debt burden relative to income |
| `credit_stress` | `dti × revol_util` | Interaction of utilisation and indebtedness |

### Categorical features
`home_ownership`, `purpose` — encoded natively by LightGBM (no one-hot encoding needed)

---

## Outputs

All files are written to `outputs/`:

| File | Description |
|------|-------------|
| `dashboard.png` | 2×2 panel: score distribution, profit curves, confusion matrix, SHAP bar chart |
| `shap_summary.png` | SHAP beeswarm plot showing feature impact direction and magnitude |
| `cohort_analysis.csv` | Count and observed default rate per risk tier |
| `metrics.json` | OOT ROC-AUC, Average Precision, and Brier Score |
| `profit_scenarios.json` | Optimal threshold and max portfolio profit for each scenario |

---

## Model Pipeline

```
Raw CSV
  └─ Filter (Fully Paid / Charged Off)
       └─ Chronological split (80% train | 20% OOT test)
            └─ Feature engineering (clean + ratio features)
                 └─ Train/val split (90% / 10% of train, stratified)
                      └─ LightGBM with early stopping (AUC)
                           └─ Isotonic calibration (3-fold CV)
                                └─ OOT evaluation + outputs
```

---

## License

This project is released for educational and research purposes. The LendingClub dataset is subject to Kaggle's terms of use.
