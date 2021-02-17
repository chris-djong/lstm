from connection import get_stock_data
from lstm import StockPredictor
import numpy as np 
import pandas as pd

# Create the lstm model
stock_predictor = StockPredictor()

# Obtain the stock data and feed it to the neural net
stock_data = get_stock_data([104])
stock_data = stock_data.set_index('date')
stock_data.index = pd.to_datetime(stock_data.index)
stock_data = stock_data.sort_index().resample('D').mean()

# Split data in case we have more than 4 missing days
print(stock_data.id.isnull().astype(int).groupby(stock_data.id.notnull().astype(int).cumsum()).sum())

