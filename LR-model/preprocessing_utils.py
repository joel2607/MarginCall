import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)


def linear_regression(x, y):
    """
    performs linear regression given x and y. outputs regression coefficient
    """
    #fit linear regression
    lr = LinearRegression()
    lr.fit(x, y)
    
    return lr.coef_[0][0]

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            
            y = df['close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value
            
    return df

def normalized_values(high, low, close):
    """
    normalize the price between 0 and 1.
    """
    #epsilon to avoid deletion by 0
    epsilon = 10e-10
    
    #subtract the lows
    high = high - low
    close = close - low
    return close/(high + epsilon)

def analyze_dataset(df):
    analysis = {
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
    }
    
    # Add basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    unique_str_values = {}
    
    for column in df.columns:
        if column not in numeric_cols:
            unique_str_values[column] = df[column].value_counts()
    
    analysis['non_numeric_cols'] = unique_str_values
    
    return analysis

def clean_dataset(df):
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    def handle_missing_values(df):
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            mode = df[col].mode()[0]
            if mode == 'nan':
                print(f'{col} has been removed due to nan being the mode.' )
                df.drop(col, axis=1, inplace=True )
            else:
                df[col] = df[col].fillna(mode).replace('nan', mode).replace('nota', mode)
            
        return df
    
    
    # Handle outliers using IQR method
    def handle_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.05)
            Q3 = df[column].quantile(0.95)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df
    
    
    # Standardize text data
    def clean_text_columns(df):
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            # Convert to string type, remove whitespace, make lower
            df[col] = df[col].astype(str).str.strip().str.lower()
        

        return df
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = handle_outliers(df_clean, numeric_columns)

    df_clean = clean_text_columns(df_clean)
    
    # 5. Handle missing values after other cleaning steps
    df_clean = handle_missing_values(df_clean)

    df_clean = df_clean.drop_duplicates()
    
    
    return df_clean
    
def extract_features(data, n = 10):
    data['normalized_value'] = data.apply(lambda x: print(x) normalized_values(x.high, x.low, x.close), axis = 1)
    
    #column with local minima and maxima
    data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
    data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order = n)[0]]['close']

    #idx with mins and max
    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    return data, idx_with_mins, idx_with_maxs

def create_train_data(data, n = 10):

    #get data to a dataframe
    data, idxs_with_mins, idxs_with_maxs = extract_features(data, n)
    
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(5, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(10, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(20, data, list(idxs_with_mins) + list(idxs_with_maxs))
  
    _data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop = True)
    
    #create a dummy variable for local_min (0) and max (1)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_.loc_max]
    
    #columns of interest
    cols_of_interest = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
    _data_ = _data_[cols_of_interest]
    
    return _data_.dropna(axis = 0)

def create_test_data_lr(data, n = 10):
    """
    this function create test data sample for logistic regression model
    """
    #get data to a dataframe
    data = extract_features(data, n)
    idxs = np.arange(0, len(data))
    
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
    
    cols = ['close', 'volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    data = data[cols]

    return data.dropna(axis = 0)

def predict_trend(data, n = 10):

    #get data to a dataframe
    data = extract_features(data, n)
    
    idxs = np.arange(0, len(data))
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
        
    #create a column for predicted value
    data['pred'] = np.nan

    #get data
    cols = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    x = data[cols]

    #scale the x data
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    for i in range(x.shape[0]):
        
        try:
            data['pred'][i] = _model_.predict(x[i, :])

        except:
            data['pred'][i] = np.nan

    return data

