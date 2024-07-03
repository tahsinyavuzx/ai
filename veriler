import ccxt
import pandas as pd

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h')
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
