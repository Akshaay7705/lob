# LOB Mid-Price Predictor

A Python pipeline that forecasts short-horizon mid-price movement from Limit Order Book (LOB) data. It covers book reconstruction, feature engineering, labeling, baseline and deep models, and evaluation that accounts for realistic trading costs.

## Overview

The **mid-price** is `(best_bid + best_ask) / 2`. Given a window of recent order book snapshots, the model predicts whether the mid-price will move **up**, **down**, or stay **flat** over a fixed future horizon (for example, the next 10 to 100 events).

The project is built as a research pipeline. The aim is to measure whether a signal is real and survives costs, not just to maximize a headline accuracy number.

## Features

- Order book reconstruction from raw message/event data
- Microstructure features: multi-level imbalance, spread, depth ratios, order flow imbalance (OFI), microprice
- Smoothed-horizon labeling with a configurable threshold for the flat class
- Baselines: logistic regression and gradient boosting (LightGBM/XGBoost)
- Deep models: LSTM and DeepLOB-style CNN + LSTM (PyTorch)
- Chronological and walk-forward validation to prevent leakage
- Evaluation with per-class F1, Matthews correlation coefficient, and a simple cost-aware backtest

## Project Structure

```
lob-midprice-predictor/
├── data/
│   ├── raw/                 # Raw LOB / message files (not committed)
│   └── processed/           # Reconstructed books, features, labels
├── src/
│   ├── data_loader.py       # Parsing and book reconstruction
│   ├── features.py          # Imbalance, OFI, spread, depth features
│   ├── labels.py            # Smoothed mid-price labeling
│   ├── models/
│   │   ├── baselines.py     # Logistic regression, LightGBM
│   │   ├── lstm.py          # LSTM model
│   │   └── deeplob.py       # CNN + LSTM architecture
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Metrics and walk-forward evaluation
│   └── backtest.py          # Cost-aware signal backtest
├── notebooks/               # Exploration and result analysis
├── configs/                 # Experiment configs (YAML)
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/lob-midprice-predictor.git
cd lob-midprice-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Core dependencies:** `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `torch`, `matplotlib`, `pyyaml`

## Data

The pipeline is designed to work with:

- **FI-2010**: public benchmark LOB dataset
- **LOBSTER**: NASDAQ order book reconstructions
- **Crypto exchange feeds**: for example, Binance or Coinbase L2/L3 data

Place raw files in `data/raw/` and point the config to them. Datasets are not included in this repository; follow each provider's license and download terms.

## Usage

**1. Build features and labels**

```bash
python -m src.features --config configs/default.yaml
python -m src.labels --config configs/default.yaml
```

**2. Train a model**

```bash
python -m src.train --config configs/deeplob.yaml
```

**3. Evaluate**

```bash
python -m src.evaluate --config configs/deeplob.yaml --walk-forward
```

**4. Run the cost-aware backtest**

```bash
python -m src.backtest --config configs/deeplob.yaml --fee-bps 0.5
```

## Methodology

**Labeling.** Let `m_t` be the mid-price at time `t`, and let `m̄_t` be the mean of the next `k` mid-prices. The relative change is `l_t = (m̄_t − m_t) / m_t`. The label is:

- **Up** if `l_t > θ`
- **Down** if `l_t < −θ`
- **Flat** otherwise

**Validation.** All splits are strictly chronological. Normalization statistics are computed on the training window only. Walk-forward validation is used to test stability over time.

**Class imbalance.** Flat dominates in most regimes. The pipeline supports class weighting and threshold tuning, and reports per-class metrics rather than accuracy alone.

## Results

| Model               | Horizon | Accuracy | Macro F1 | MCC |
|---------------------|---------|----------|----------|-----|
| Logistic Regression | k=10    | TBD      | TBD      | TBD |
| LightGBM            | k=10    | TBD      | TBD      | TBD |
| LSTM                | k=10    | TBD      | TBD      | TBD |
| DeepLOB             | k=10    | TBD      | TBD      | TBD |

> Fill in after running experiments. Report results both before and after transaction costs.

## Limitations

- Predictive accuracy on mid-price direction does not imply profitability. Spread, fees, latency, and queue position can erase the edge.
- Financial data is non-stationary, so models can degrade when market regimes shift.
- Backtests here are simplified and do not model market impact or full queue dynamics.

## Roadmap

- [ ] Reproduce a DeepLOB baseline on FI-2010
- [ ] Add Transformer-based model
- [ ] Add queue-position-aware execution simulation
- [ ] Expand to multi-asset evaluation

## References

- Zhang, Zohren, Roberts. *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books* (2019)
- Ntakaris et al. *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods* (FI-2010)
- Cont, Kukanov, Stoikov. *The Price Impact of Order Book Events* (2014)

## License

MIT License. See `LICENSE` for details.
