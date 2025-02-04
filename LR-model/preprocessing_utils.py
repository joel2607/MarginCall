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
    n day regression with proper index handling.
    
    Parameters:
    n (int): Number of days for regression window
    df (pandas.DataFrame): DataFrame with price data
    idxs (list): List of indices to calculate regression for
    
    Returns:
    pandas.DataFrame: DataFrame with new regression column
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Variable name for the regression column
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan
    
    # Filter idxs to only include those present in df
    valid_idxs = [idx for idx in idxs if idx < len(df)]
    
    for i, idx in enumerate(valid_idxs):
        if i == 0:
            # For first point, use the slope between first two points if available
            if len(df) > 1:
                y = df['close'].iloc[0:2].to_numpy()
                x = np.array([0, 1])
                y = y.reshape(y.shape[0], 1)
                x = x.reshape(x.shape[0], 1)
                coef = linear_regression(x, y)
                df.iloc[idx, df.columns.get_loc(_varname_)] = coef
            else:
                df.iloc[idx, df.columns.get_loc(_varname_)] = 0
                
        elif i < n:
            # For points before n, use available historical data
            y = df['close'].iloc[0:i+1].to_numpy()
            x = np.arange(0, i+1)
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            coef = linear_regression(x, y)
            df.iloc[idx, df.columns.get_loc(_varname_)] = coef
            
        else:
            # Normal case - use full n-day window
            y = df['close'].iloc[idx-n:idx].to_numpy()
            x = np.arange(0, n)
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            coef = linear_regression(x, y)
            df.iloc[idx, df.columns.get_loc(_varname_)] = coef
    
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
    data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis = 1)
    
    #column with local minima and maxima
    data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
    data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order = n)[0]]['close']

    #idx with mins and max
    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    return data, idx_with_mins, idx_with_maxs

def create_train_data(data, n = 10):

    print("Initial data shape:", data.shape)
    
    # Get data to a dataframe
    data, idxs_with_mins, idxs_with_maxs = extract_features(data, n)
    
    # Print shapes of mins and maxs
    print("Indices with mins:", len(idxs_with_mins))
    print("Indices with maxs:", len(idxs_with_maxs))
    
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

def create_test_data_lr(data, _model_, n = 10):
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

def _threshold(predictions, threshold):

        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]

        return np.array(prob_thresholded)

def predict_trend(data, model, n=10):
    """
    Predicts trends in financial data using the provided model while maintaining DataFrame structure.
    
    Parameters:
    data (pandas.DataFrame): Input DataFrame with financial data
    model: Trained prediction model
    n (int): Window size for feature extraction
    
    Returns:
    pandas.DataFrame: DataFrame with predictions
    """
    # Create a copy to avoid modifying the original
    data_processed = data.copy()
    
    # Extract features while maintaining original index
    data_processed, _, _ = extract_features(data_processed, n)
    
    # Calculate regression features
    idxs = np.arange(len(data_processed))
    data_processed = n_day_regression(3, data_processed, idxs)
    data_processed = n_day_regression(5, data_processed, idxs)
    data_processed = n_day_regression(10, data_processed, idxs)
    data_processed = n_day_regression(20, data_processed, idxs)
    
    # Select needed columns
    cols = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    x = data_processed[cols]
    
    # Handle any remaining missing values
    # x = x.fillna(method='ffill').fillna(method='bfill')
    
    # Scale the features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Initialize predictions column
    data_processed['pred'] = np.nan
    data_processed['threshold_pred'] = np.nan
    
    # Make predictions using the DataFrame's existing index
    for idx in range(len(x_scaled)):
        x_set = x_scaled[idx, :].reshape(1, -1)
        try:
            data_processed.iloc[idx, data_processed.columns.get_loc('pred')] = model.predict(x_set)
            data_processed.iloc[idx, data_processed.columns.get_loc('threshold_pred')] = _threshold(model._predict_proba_lr(x_set), 0.98)
        except Exception as err:
            print(f"Exception occurred at index {idx}")
            print(err)
            data_processed.iloc[idx, data_processed.columns.get_loc('pred')] = np.nan
            data_processed.iloc[idx, data_processed.columns.get_loc('threshold_pred')] = np.nan
    
    return data_processed