# Appliance Load Forecasting

This project contains a Jupyter Notebook for forecasting appliance electricity load using statistical time-series models and hybrid deep-learning residual models.

## Overview

The notebook focuses on hourly appliance load forecasting for a fridge dataset. It compares standalone statistical models with hybrid approaches that combine linear forecasts and LSTM-based residual correction.

## Data Source

The data used in this project is taken from the UK-DALE dataset, specifically House 1 appliance electricity data.
## Models

- Single ARIMA with AIC, BIC, and MML model selection
- Hybrid ARIMA-LSTM using residual forecasting
- Single ARFIMA with fractional differencing
- Hybrid ARFIMA-LSTM using residual forecasting

## Notebook Structure

The notebook is organized into clear stages:

1. Environment setup and imports
2. ARIMA/ARFIMA helper functions
3. MML scoring utilities
4. Residual modelling with ANN/LSTM
5. Hyperparameter tuning
6. Data preprocessing and rolling-window setup
7. Single and hybrid model experiments
8. Checkpoint saving and result comparison

## Files

- `Appliance-load-electricty.ipynb` - Main notebook for data preprocessing, modelling, rolling forecasts, and result saving

## Requirements

The notebook is designed to run in Google Colab and uses:

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- statsmodels
- TensorFlow/Keras
- tqdm

## Output

Experiment results are saved into a master checkpoint file (`master_results.pkl`) by rolling forecast date and model name. This allows the notebook to resume previous runs and compare model performance using RMSE and MAPE.

## Notes

The notebook uses Google Drive paths for data and checkpoints. Update the paths in the configuration section before running the notebook in a new environment.
