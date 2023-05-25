import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import logging
from os import makedirs
import threading
import kaggle

logging.info("*******Starting preprocessing/feature engineering script*******")
logging.info("Downloading dataset")
kaggle.api.authenticate()
kaggle.api.dataset_download_files("jacksoncrow/stock-market-dataset", "archive", unzip=True)

os.makedirs("engineered_data", exist_ok=True)
os.makedirs("extracted_data", exist_ok=True)

logging.basicConfig(filename="logfile.log", level=logging.INFO, filemode='w')

meta_data = pd.read_csv("archive/symbols_valid_meta.csv")
mapping = {x[0]: x[1] for x in meta_data[["Symbol", "Security Name"]].to_numpy()}

stocks = [x[:-4] for x in os.listdir("archive/stocks/")]
etfs = [x[:-4] for x in os.listdir("archive/etfs/")]

def etf_processing_and_engineering():
    data_list_original = []
    data_list_feature_eng = []

    logging.info("Extracting ETF data, and engineering new features")
    for etf in tqdm(etfs):
        data = pd.read_csv(f"archive/etfs/{etf}.csv", parse_dates=["Date"], infer_datetime_format=True)
        data["Symbol"] = etf
        try:
            data["Security Name"] = mapping[etf]
        except:
            logging.error(f"No mapping for {etf}")
            continue
    
        data_list_original.append(data)
        data_copy = data.copy(deep = True).set_index("Date").sort_index().dropna()
        data_copy['vol_moving_avg'] = data_copy['Volume'].rolling("30d").mean()
        data_copy['adj_close_rolling_med'] = data_copy['Adj Close'].rolling("30d").median()

        data_list_feature_eng.append(data_copy)
    
    etf_data = pd.concat(data_list_original)
    etf_data_feature_eng = pd.concat(data_list_feature_eng)
    etf_data.to_parquet("extracted_data/etf_data.parquet")
    etf_data_feature_eng.to_parquet("engineered_data/etf_data.parquet")


def stock_processing_and_engineering():
    data_list_original = []
    data_list_feature_eng = []

    logging.info("Extracting Stock data, and engineering new features")
    for stock in tqdm(stocks):
        data = pd.read_csv(f"archive/stocks/{stock}.csv", parse_dates=["Date"], infer_datetime_format=True)
        data["Symbol"] = stock
        try:
            data["Security Name"] = mapping[stock]
        except:
            logging.error(f"No mapping for {stock}")
            continue
    
        data_list_original.append(data)
        data_copy = data.copy(deep = True).set_index("Date").sort_index().dropna()
        data_copy['vol_moving_avg'] = data_copy['Volume'].rolling("30d").mean()
        data_copy['adj_close_rolling_med'] = data_copy['Adj Close'].rolling("30d").median()

        data_list_feature_eng.append(data_copy)

    stock_data = pd.concat(data_list_original)
    stock_data_feature_eng = pd.concat(data_list_feature_eng)
    stock_data.to_parquet("extracted_data/stock_data.parquet")
    stock_data_feature_eng.to_parquet("engineered_data/stock_data.parquet")

def preprocess():
    t1 = threading.Thread(target=etf_processing_and_engineering)
    t2 = threading.Thread(target=stock_processing_and_engineering)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    logging.info("*******Ending preprocessing/feature engineering script*******")


preprocess()