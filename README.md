# NVDA Volatility Forecasting with Deep Learning

This repository contains the full implementation of my final-year dissertation investigating **intraday and intrahour volatility regime classification** for NVIDIA (NVDA) using deep learning architectures (Vanilla RNN, LSTM, GRU) and a GARCH(1,1) benchmark.

The system is structured around two prediction horizons:

1. **Intrahour** – 1-minute bar data, classifying volatility regimes within the trading day.
2. **Intraday** – daily-level features including VIX, classifying next-day volatility regimes.

Key results:
| Model | Horizon | Accuracy | AUC |
|---|---|---|---|
| GRU | Intrahour | 70.7% | 0.773 |
| GRU | Intraday | 65.9% | 0.777 |
| GARCH(1,1) | Intraday | 66.8% | 0.480 |

---

## Repository Structure

```text
Dissertation/
├── Dataset Construction/
│   ├── Data Cleaning/              # MAD-based outlier removal + diagnostics
│   ├── Intrahour Construction/     # 1-min bar dataset construction notebook + CSV
│   ├── Intraday Construction/      # Daily dataset construction with VIX notebook + CSV
│   ├── GARCH/
│   │   ├── GARCH Code/             # GARCH(1,1) benchmark notebook and forecast outputs
│   │   └── GARCH Docs/             # Diagnostic plots (ACF, ROC)
│   ├── Baseline and final comparison/  # Final LSTM and GRU model notebooks
│   ├── ML Models Comparison/       # Intrahour RNN/LSTM/GRU comparison notebook
│   ├── Data Ordering Test/         # Ordered vs shuffled experiment notebook
│   ├── Feature Correlation/        # Intraday CCF/feature analysis notebook
│   └── SHAP Analysis/              # SHAP importance and time-series plots
├── Images/                         # Supporting visualisations (vol clustering, delta, bid-ask, etc.)
└── Raw Data Files.zip              # Raw 1-min NVDA tick data
```

---

## Prerequisites

Python 3.10+ is recommended. Install dependencies via:

```bash
pip install -r requirements.txt
```

Core packages used across notebooks:

- `pandas`, `numpy` – data manipulation
- `tensorflow` / `keras` – LSTM, GRU, Vanilla RNN models
- `arch` – GARCH(1,1) implementation
- `shap` – SHAP feature attribution
- `scikit-learn` – metrics, train/test splitting, preprocessing
- `matplotlib`, `seaborn` – visualisation

---

## Running the Pipeline

All notebooks should be run in the order below. Each notebook saves its outputs (CSVs, model weights, plots) in its own directory.

### 1. Data Cleaning

Cleans raw 1-minute NVDA price data using a 2× MAD threshold for outlier removal.

```
Dataset Construction/Data Cleaning/NVDA_Intrahour_Data_Cleaning.ipynb
```

Outputs: cleaned volatility and return-squared histograms.

---

### 2. Dataset Construction

**Intrahour dataset:**
```
Dataset Construction/Intrahour Construction/NVDA_Intrahour_Volatility_Dataset_Construction.ipynb
```
Output: `NVDA_Intrahour_Volatility_Dataset.csv`

**Intraday dataset (with VIX):**
```
Dataset Construction/Intraday Construction/NVDA_Intraday_Volatility_Dataset_Construction_VIX.ipynb
```
Output: `NVDA_Intraday_Volatility_Dataset_VIX.csv`

---

### 3. Feature Analysis

Cross-correlation analysis to validate VIX as the leading intraday feature.

```
Dataset Construction/Feature Correlation/NVDA Intraday Feature Analysis.ipynb
```

---

### 4. Data Ordering Test

Validates that temporal ordering is preserved (ordered vs. shuffled experiment).

```
Dataset Construction/Data Ordering Test/NVDA_Intrahour_LSTM_Ordered_vs_Shuffled.ipynb
```

---

### 5. GARCH Benchmark

Expanding-window GARCH(1,1) classification used as the econometric baseline.

```
Dataset Construction/GARCH/GARCH Code/NVDA_GARCH_Benchmark.ipynb
```

Outputs: forecast CSV, ROC curve, diagnostic plots.

---

### 6. Model Comparison (Intrahour)

Trains and compares Vanilla RNN, LSTM, and GRU on the intrahour dataset.

```
Dataset Construction/ML Models Comparison/NVDA_Intrahour_LSTM_GRU_RNN_Metrics.ipynb
```

---

### 7. Final Models (Intraday)

Trains the final LSTM and GRU models on the intraday VIX dataset.

```
Dataset Construction/Baseline and final comparison/NVDA Intraday LSTM.ipynb
Dataset Construction/Baseline and final comparison/NVDA Intraday GRU.ipynb
```

Saved model weights: `NVDA_LSTM_VIX_best_model.keras`, `NVDA_GRU_VIX_best_model.keras`

---

### 8. SHAP Analysis

SHAP feature attribution on the final LSTM model. 

```
Dataset Construction/SHAP Analysis/NVDA_LSTM_VIX_SHAP.ipynb
```

Outputs: Feature importance bar chart, SHAP-over-time plot.

---

## Datasets

| File | Description |
|---|---|
| `NVDA_Intrahour_Volatility_Dataset.csv` | Cleaned 1-min bar features for intrahour modelling |
| `NVDA_Intraday_Volatility_Dataset_VIX.csv` | Daily features including VIX for intraday modelling |
| `NVDA_GARCH_Forecasts.csv` | GARCH(1,1) expanding-window volatility forecast outputs |

Raw 1-minute tick data is excluded from this repository due to file size constraints.
