import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from joblib import load, dump

logging.basicConfig(filename="logfile.log", level=logging.INFO, filemode='a')
logging.info("*******Starting ML script*******")
logging.info("Reading data from disk")

stock_data = pd.read_parquet("engineered_data/stock_data.parquet")
etf_data = pd.read_parquet("engineered_data/etf_data.parquet")

data = pd.concat([stock_data, etf_data])

data.dropna(inplace=True)

features = ['vol_moving_avg', 'adj_close_rolling_med']
target = 'Volume'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save():
    model = AdaBoostRegressor(n_estimators=100, random_state=42)
    logging.info("Training model")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Mean absolute error = {mae}")
    logging.info(f"Mean squared error = {mse}")
    logging.info(f"R2 score = {r2}")

    logging.info("Saving model to disk")
    dump(model, "model.clf")
    logging.info("*******Ending ML script*******")

train_and_save()