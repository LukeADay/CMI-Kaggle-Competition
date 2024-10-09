#%%
import numpy as np
import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.optimize import minimize
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.feature_selection import RFECV
import warnings
#from lightgbm import LGBMRegressor
#from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
#%%
import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv(r"../data/child-mind-institute-problematic-internet-use/train.csv")
test = pd.read_csv(r"../data/child-mind-institute-problematic-internet-use/test.csv")
sample = pd.read_csv(r"../data/child-mind-institute-problematic-internet-use/sample_submission.csv")

# %%

class DataHandler:
    """
    Class that handles data loading and preparation
    
    """

    def __init__(self):
        # Load tabular data
        train, test, sample = self.load_tabular_data()
        
        # Load time-series
        train_ts = self.load_time_series(r"../data/child-mind-institute-problematic-internet-use/series_train.parquet")
        test_ts = self.load_time_series(r"../data/child-mind-institute-problematic-internet-use/series_test.parquet")

        # Grab column names
        time_series_cols = train_ts.columns.tolist()
        time_series_cols.remove("id")

        # Join the tabular and time-series data together
        train = pd.merge(train, train_ts, how="left", on='id')
        test = pd.merge(test, test_ts, how="left", on='id')

        train = train.drop('id', axis=1)
        test = test.drop('id', axis=1)

        train = train[train["sii"].notna()] 
        self.train = train
        self.test = test
        self.sample = sample

    def load_tabular_data(self):
        train = pd.read_csv(r"../data/child-mind-institute-problematic-internet-use/train.csv")
        test = pd.read_csv(r"../data/child-mind-institute-problematic-internet-use/test.csv")
        sample = pd.read_csv(r"../data/child-mind-institute-problematic-internet-use/sample_submission.csv")
        return train, test, sample
    
    def time_features(self, df):
        # Convert time_of_day to hours
        df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
        # Basic features 
        features = [
            df["non-wear_flag"].mean(),
            df["battery_voltage"].mean(),
            df["battery_voltage"].diff().mean(),
            df["relative_date_PCIAT"].tail(1).values[0]
        ]
        
    #     df = df[df["non-wear_flag"] == 0]
        # Define conditions for night, day, and no mask (full data)
        night = ((df["hours"] >= 22) | (df["hours"] <= 5))
        day = ((df["hours"] <= 20) & (df["hours"] >= 7))
        no_mask = np.ones(len(df), dtype=bool)
        wear = ~(df["non-wear_flag"].astype(bool))
        
        # List of columns of interest and masks
        keys = ["enmo", "anglez", "light"]
        masks = [no_mask, night, day, wear]
        
        # Helper function for feature extraction
        def extract_stats(data):
            return [
                data.mean(), 
                data.std(), 
                data.max(), 
                data.min(), 
                data.kurtosis(), 
                data.skew(), 
                data.diff().mean(), 
                data.diff().std(), 
                data.diff().quantile(0.9), 
                data.diff().quantile(0.1)
            ]
        
        # Iterate over keys and masks to generate the statistics
        for key in keys:
            for mask in masks:
                filtered_data = df.loc[mask, key]
                features.extend(extract_stats(filtered_data))

        return features

    # Code for parallelized computation of time series data from: Sheikh Muhammad Abdullah 
    # https://www.kaggle.com/code/abdmental01/cmi-best-single-model
    def process_file(self, filename, dirname):
        # Process file and extract time features
        df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
        df.drop('step', axis=1, inplace=True)
        return self.time_features(df), filename.split('=')[1]

    def load_time_series(self, dirname):
        # List of files or subdirectories in the given directory
        ids = [fname for fname in os.listdir(dirname) if not fname.startswith('.')]
        
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda fname: self.process_file(fname, dirname), ids), total=len(ids)))
        
        stats, indexes = zip(*results)
        
        df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
        df['id'] = indexes
        
        return df

# Initialise object to grab processed test and train datasets
data_obj = DataHandler()
train = data_obj.train # Grab the train set
test = data_obj.test # Grab the test set

# Remove columns not in the test set and set target column
exclude = [col for col in train.columns if col not in test.columns]
y = "PCIAT-PCIAT_Total" # Score, target for the model
target = "sii" # Index, target of the competition
features = [f for f in train.columns if f not in exclude]

# Grab categorical columns
cat_c = train.select_dtypes(include = 'object').columns
