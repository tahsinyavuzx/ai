import ccxt
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# Bitcoin fiyat verilerini toplama
def fetch_price_data():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h')
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Haber verilerini toplama
def fetch_news():
    url = 'https://news.ycombinator.com/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    titles = soup.find_all('a', class_='storylink')
    news = [title.get_text() for title in titles]
    return news

# Haber analizini gerçekleştirme
def analyze_sentiment(news):
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(article)['compound'] for article in news]
    avg_sentiment = np.mean(scores)
    return avg_sentiment

# LSTM modelini oluşturma
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model eğitimi
def train_lstm_model(model, train_data, epochs=25, batch_size=32):
    X_train, y_train = create_dataset(train_data)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Veri ölçeklendirme
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Veri seti oluşturma
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Ana döngü
def main():
    df = fetch_price_data()
    close_prices = df['close'].values.reshape(-1, 1)
    scaled_data, scaler = scale_data(close_prices)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    model = create_lstm_model((60, 1))
    train_lstm_model(model, train_data)

    look_back = 60
    while True:
        news = fetch_news()
        sentiment = analyze_sentiment(news)
        
        df = fetch_price_data()
        close_prices = df['close'].values.reshape(-1, 1)
        scaled_data, scaler = scale_data(close_prices)

        last_look_back_data = scaled_data[-look_back:]
        X_test = np.array([last_look_back_data])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)
        current_price = close_prices[-1][0]

        if sentiment > 0 and prediction > current_price:
            print("Long pozisyon aç!")
        elif sentiment < 0 and prediction < current_price:
            print("Short pozisyon aç!")
        else:
            print("Pozisyon açma!")

        # Yeni verilerle modeli yeniden eğit
        train_lstm_model(model, scaled_data)
        
        time.sleep(3600)  # 1 saat bekleme

if __name__ == "__main__":
    main()
