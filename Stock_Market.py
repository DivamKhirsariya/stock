# Import required libraries
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pyportfolioopt import EfficientFrontier, risk_models, expected_returns

# Configuration
TICKER = "AAPL"
START_DATE = "2010-01-02"
END_DATE = "2023-12-31"
LOOKBACK_WINDOW = 60  # Using 60 trading days (~3 months) as historical context
PREDICTION_HORIZON = 10  # Predict next 10 days

# Advanced Feature Engineering
def add_technical_indicators(df):
    # Relative Strength Index
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['20MA'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['20MA'] - (df['20STD'] * 2)
    
    return df.dropna()

# Transformer Model Architecture
def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding Layer
    x = layers.Dense(64)(inputs)  # Embedding dimension
    x = PositionalEncoding()(x)
    
    # Transformer Block
    x = TransformerBlock(embed_dim=64, num_heads=2, ff_dim=64)(x)
    x = TransformerBlock(embed_dim=64, num_heads=2, ff_dim=64)(x)
    
    # Output Layer
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(PREDICTION_HORIZON)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
                 loss='mse', 
                 metrics=['mae'])
    return model

# Custom Transformer Components
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        seq_length = tf.shape(x)[1]
        position = tf.range(seq_length, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
        div_term = tf.exp(tf.range(0, 64, 2, dtype=tf.float32) * (-np.log(10000.0) / 64))
        pe = np.zeros((1, seq_length, 64))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        return x + tf.cast(pe, dtype=x.dtype)

# Data Pipeline
def create_dataset(data, lookback=LOOKBACK_WINDOW):
    X, y = [], []
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    for i in range(lookback, len(data)-PREDICTION_HORIZON):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i:i+PREDICTION_HORIZON, 0])  # Predict Close price
        
    return np.array(X), np.array(y), scaler

# Main Execution
if __name__ == "__main__":
    # 1. Data Acquisition
    df = yf.download(TICKER, START_DATE, END_DATE)
    df = add_technical_indicators(df)
    features = df[['Close', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band']]
    
    # 2. Data Preparation
    X, y, scaler = create_dataset(features.values)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 3. Model Training
    model = build_transformer_model((LOOKBACK_WINDOW, X_train.shape[2]))
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=100, 
                        batch_size=32, 
                        callbacks=[early_stopping])
    
    # 4. Prediction and Evaluation
    predictions = model.predict(X_val)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_val.reshape(-1,1)).reshape(y_val.shape)
    
    # Calculate metrics
    mse = mean_squared_error(actual.flatten(), predictions.flatten())
    mae = mean_absolute_error(actual.flatten(), predictions.flatten())
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Visualize predictions
    plt.figure(figsize=(15,6))
    plt.plot(actual[-100:].flatten(), label='Actual')
    plt.plot(predictions[-100:].flatten(), label='Predicted')
    plt.title("Stock Price Prediction using Transformer Network")
    plt.legend()
    plt.show()
    
    # 5. Risk Assessment with Monte Carlo Simulation
    last_sequence = X_val[-1][np.newaxis, ...]
    mc_predictions = []
    for _ in range(1000):
        pred = model.predict(last_sequence)
        mc_predictions.append(scaler.inverse_transform(pred)[0])
    mc_predictions = np.array(mc_predictions)
    
    # Calculate Value at Risk (VaR)
    final_prices = mc_predictions[:, -1]
    var_95 = np.percentile(final_prices, 5)
    print(f"95% VaR: {var_95:.2f}")
    
    # 6. Portfolio Optimization using Efficient Frontier
    mu = expected_returns.mean_historical_return(df['Close'])
    S = risk_models.sample_cov(df['Close'])
    
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()
    print("Optimal Portfolio Allocation:")
    print(weights)