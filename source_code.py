
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Local Dataset (AAPL_live.csv)
def load_local_data(filepath='AAPL_live.csv'):
    df = pd.read_csv('AAPL_live.csv', parse_dates=['Date'], index_col='Date')
    df = df[['Close']].dropna()
    return df

# 2. Plot
def plot_data(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df, label='Close Price')
    plt.title('AAPL Live Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()

# 3. ADF Stationarity Test
def is_stationary(df):
    result = adfuller(df['Close'])
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    return result[1] < 0.05

# 4. Prepare for LSTM
def prepare_lstm_data(df, window=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, scaled_data

# 5. Build LSTM Model
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 6. Forecast
def forecast_lstm(days_to_predict=10):
    df = load_local_data()
    X_train, y_train, scaler, scaled_data = prepare_lstm_data(df)

    model = build_lstm((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    last_window = scaled_data[-60:].reshape(1, 60, 1)
    predictions = []

    for _ in range(days_to_predict):
        pred = model.predict(last_window, verbose=0)
        predictions.append(pred[0][0])
        last_window = np.append(last_window[:, 1:, :], [[[pred[0][0]]]], axis=1)

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': predicted_prices
    })
    result.set_index('Date', inplace=True)
    return result

# 7. Gradio Interface
interface = gr.Interface(
    fn=forecast_lstm,
    inputs=gr.Slider(minimum=5, maximum=30, step=1, value=10, label="Days to Forecast"),
    outputs=gr.Dataframe(label="Predicted Stock Prices"),
    title="AAPL Live Stock Price Forecasting (LSTM)",
    description="Predict future AAPL stock prices using LSTM trained on AAPL_live dataset"
)

# The code inside the if __name__ == "__main__": block needs to be indented
if __name__ == "__main__":
    interface.launch()