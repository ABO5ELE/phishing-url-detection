# preprocessing.py
# this file loads and prepares the dataset

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # load the csv file
    df = pd.read_csv(file_path)
    return df

def prepare_data(df):
    # remove url column because its text
    X = df.drop(columns=['url', 'status'])
    y = df['status']
    
    # convert labels to numbers
    y = y.map({'legitimate': 0, 'phishing': 1})
    
    return X, y

def split_data(X, y):
    # split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
