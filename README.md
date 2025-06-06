# EGX-Stock-Prediction

## Overview
This project leverages deep learning techniques to forecast closing prices of stocks listed in the Egyptian Stock Exchange (EGX), specifically targeting EGX 30 and EGX 70 indices. Using a Long Short-Term Memory (LSTM) model built with TensorFlow and Keras, the project aims to capture time-dependent patterns in stock data for more accurate predictions. The implementation supports small-scale investors, financial analysts, and educational exploration of market forecasting.

## Objectives
- Analyze historical EGX stock data to forecast the “Close” price.
- Use LSTM networks to model time series dependencies.
- Prepare the data for robust training and evaluation.
- Generate predictions for unlabeled future entries.

## Dataset
The project uses three CSV files:
- `train.csv`: Labeled historical data with columns: `Date`, `Price` (Close), `Open`, `High`, `Low`, and `Vol.`
- `val.csv`: Separate validation data with the same structure as `train.csv`
- `test_without_label.csv`: Contains `ID`, `Date`, `Open`, `High`, `Low`, and `Vol.` but no `Price` label.

## Key Components

### 1. Data Preprocessing
- Volume formatting: Converts `1.2M`, `876.9K`, etc., into numeric form.
- Normalization with `MinMaxScaler` for numerical stability.
- Sliding window generation to create time-series input sequences for LSTM training.

### 2. Model Architecture
- 2 stacked LSTM layers for temporal pattern recognition
- Dense output layer for final price prediction
- `Adam` optimizer and `Mean Squared Error` loss function
- Includes `EarlyStopping` to prevent overfitting

### 3. Training and Evaluation
- Trains on `train.csv`, validates on `val.csv`
- Plots training vs. validation loss
- Predicts and visualizes on unseen data

### 4. Inference and Submission
- Applies trained model on `test_without_label.csv`
- Saves predicted prices in a CSV format for review

## Technologies Used
- Python 3
- TensorFlow / Keras
- Pandas, NumPy, scikit-learn
- Matplotlib

