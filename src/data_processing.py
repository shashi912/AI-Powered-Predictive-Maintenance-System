import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    # Standardize sensor readings
    scaler = StandardScaler()
    sensor_cols = [col for col in data.columns if 'sensor' in col]
    data[sensor_cols] = scaler.fit_transform(data[sensor_cols])
    return data
