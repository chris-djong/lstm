from connection import get_stock_data, get_stocks
from lstm import LSTMModel
import numpy as np 
import pandas as pd
import datetime
from tqdm import tqdm

stocks = get_stocks()
ids = stocks['id']
newest_date = datetime.date(day=1, month=1, year=2020)

predictor = LSTMModel()
for stock_id in tqdm(ids):
    stock_data = get_stock_data([stock_id])
    stock_data = stock_data[stock_data['date']<newest_date]
    predictor.add_data(stock_data, is_train=True)

predictor.finalise_input()
predictor.train_model()



# stock_data = stock_data.set_index('date')
# stock_data.index = pd.to_datetime(stock_data.index)
# stock_data = stock_data.sort_index().resample('D').mean()

# Split data in case we have more than 4 missing days
# print(stock_data.id.isnull().astype(int).groupby(stock_data.id.notnull().astype(int).cumsum()).sum())


