# E-XAI: Evaluating Black-Box Explainable AI Frameworks for Network Intrusion Detection

## Abstract

E-XAI is a CPU-friendly research prototype for binary network-intrusion detection and the empirical evaluation of local explanations. It trains a black-box flow classifier, produces genuine SHAP and LIME explanations, measures their fidelity, stability, consistency, sparsity, runtime, and agreement, and stores repeatable experiments in SQLite. Permutation importance supplies a distinct global reference.

> The bundled synthetic flow data is for software demonstration only. Its results are not CICIDS2017/UNSW-NB15 research findings.

## Problem, objectives, and contribution

High predictive performance does not establish that an explanation is reliable. This project asks: **How reliable are different black-box XAI frameworks when explaining network-intrusion decisions?** Its contribution is a transparent evaluation layer around a NIDS—not merely SHAP/LIME charts. It supports real CSVs, uploaded data, and a deterministic fallback demo; leakage-safe preprocessing; probability-based risk; local and global XAI; quantitative comparison; persistent experiments; and report export.

## Architecture

```text
CSV / upload / demo
  → normalization, label mapping, imputation, encoding, scaling
  → stratified raw train/test split (transformer fitted on train only)
  → XGBoost (Random Forest fallback)
  → prediction + calibrated model probability thresholds
  → local SHAP / local LIME + global SHAP / permutation importance
  → E-XAI metrics + SHAP/LIME agreement + configurable quality score
  → Streamlit dashboard + SQLite history + HTML report
```

## Features and screens

The application includes Overview, Dataset, Model Performance, Intrusion Detection, Explain Prediction, XAI Evaluation, Framework Comparison, Global Analysis, Experiment History, and About / Methodology. It supports sample selection, manual flow editing, CSV upload/retraining, bounded XAI workloads, model/permutation rankings, filters, CSV history export, and HTML experiment reports.

## Installation (Python 3.11 or 3.12)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Windows reports that an application-control policy blocked a newly installed native wheel, an administrator may need to allow Python scientific packages under the applicable organizational policy. Do not weaken a managed security policy without authorization.

## Dataset setup

Three modes are supported:

1. Put authorized CICIDS2017 or UNSW-NB15 CSV files in `data/raw/`. Compatible files are automatically discovered and combined on common columns.
2. Upload a compatible labeled CSV from the Dataset screen and select **Train and activate uploaded dataset**.
3. With no real CSV present, training generates deterministic flow-like demo data and prominently labels every result as demonstration-only.

Expected labels include `Label`, `class`, `attack_cat`, `attack`, `target`, or `outcome`. `BENIGN`, `Normal`, `normal`, and numeric `0` map to benign; other nonzero/attack labels map to attack. The download helper accepts a direct, authorized CSV URL:

```powershell
python download_dataset.py https://host/path/data.csv --name unsw_nb15.csv
```

Dataset licensing and redistribution remain the user's responsibility.

## Train, run, evaluate, and test

```powershell
python train_model.py
python -m streamlit run app.py
python evaluate_xai.py
python -m pytest -q
```

From this already configured workspace, the exact interpreter form is:

```powershell
.venv\Scripts\python.exe train_model.py
.venv\Scripts\python.exe -m streamlit run app.py
.venv\Scripts\python.exe evaluate_xai.py
.venv\Scripts\python.exe -m pytest -q
```

## E-XAI metric definitions

- **Fidelity (0–1):** cosine similarity between the absolute local explanation vector and actual black-box probability changes produced by replacing each feature, one at a time, with its training-reference median.
- **Stability (0–1):** mean of aligned absolute-weight cosine similarity and Top-K Jaccard overlap across small, feature-standard-deviation-scaled perturbations that preserve the predicted class.
- **Consistency (0–1):** the same similarity measure between the event and nearest reference samples in standardized model space with the same black-box prediction.
- **Sparsity (0–1):** `1 − significant_features / total_features`, where significant means an absolute weight at least 10% of the largest local absolute weight.
- **Runtime:** raw `time.perf_counter()` wall-clock seconds for local explanation generation. Efficiency is `1 / (1 + runtime_seconds)`.
- **Agreement:** Top-K SHAP/LIME feature Jaccard plus Spearman rank correlation when at least two features overlap.

These are documented operational definitions for this prototype, not universal XAI standards.

The overall score is:

```text
Q = 0.30 Fidelity + 0.25 Stability + 0.20 Consistency
    + 0.15 Sparsity + 0.10 Efficiency
```

Weights can be changed in the UI but must be nonnegative and total 1. No explainer is predetermined to win.

## Project structure

```text
app.py                       Streamlit dashboard
train_model.py               dataset → training → saved artifacts
evaluate_xai.py              end-to-end XAI/database smoke workflow
download_dataset.py          direct CSV setup helper
modules/
  dataset_manager.py         discovery, loading, labels, demo generator
  preprocessing.py           leakage-safe reusable transformer
  model_engine.py            model selection, metrics, artifacts
  prediction_engine.py       probabilities and risk levels
  shap_engine.py             local/global SHAP
  lime_engine.py             local LIME
  permutation_engine.py      global permutation importance
  xai_evaluator.py           reliability metrics and quality score
  agreement_engine.py        Top-K/rank agreement
  database.py                parameterized SQLite operations
  visualization.py           Plotly figures
  report_generator.py        standalone HTML reports
data/                        raw, processed, and exai.db
models/                      model, preprocessor, references, metadata
sample_data/                 deterministic demo CSV
tests/                       fast unit/integration tests
```

## Preprocessing and model notes

Columns are normalized; duplicates, all-null/constant fields, unnamed columns, and common IP/flow/timestamp identifiers are removed. Infinity becomes missing data. Numeric-looking fields are safely converted; numeric data receives train-fitted median imputation and scaling; categorical data receives most-frequent imputation and unknown-safe one-hot encoding. Raw rows are split before transformer fitting to prevent test leakage. Both classes are required. The primary classifier is XGBoost; model construction falls back to a balanced Random Forest if XGBoost cannot initialize.

Risk thresholds are LOW below 35% attack probability, MEDIUM from 35% to below 60%, HIGH from 60% to below 85%, and CRITICAL from 85%. These are interface triage thresholds, not deployment-calibrated security policy.

## Limitations

- Explanation validity is conditional on the trained model and the dataset.
- LIME is sampling-based; its results can vary despite seeded construction.
- SHAP and repeated reliability checks can be computationally expensive.
- Offline flow classification does not prove real-time deployment performance.
- Global importance cannot substitute for a local event explanation.
- Explainer agreement is not evidence of causal correctness.
- Fidelity, perturbation, sparsity, and score-weight choices affect conclusions.
- The demo generator cannot support academic claims about real networks.

## Future enhancements

The module boundaries permit multiclass labels, live flow ingestion, counterfactual explainers, drift/adversarial checks, SIEM integration, MITRE ATT&CK mappings, and analyst studies without coupling them into this prototype.

## Academic disclaimer

Use real, properly licensed data and report dataset provenance, splits, model settings, metric definitions, and repeated-run uncertainty in academic work. The software demonstrates a methodology; it does not establish a universal ranking of XAI frameworks or infer attacker intent.
