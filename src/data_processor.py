import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, TIME_STEPS

def load_and_preprocess_data():
    
    df = pd.read_csv(DATA_PATH)

    df = df.drop(['Time'], axis=1)

    normal_df = df[df['Class'] == 0].drop('Class', axis=1)

    scaler = StandardScaler()
    scaled_normal_data = scaler.fit_transform(normal_df)

    X_train = create_sequences(scaled_normal_data, TIME_STEPS)

    test_df = df.drop('Class', axis=1)
    test_labels = df['Class'].values
    scaled_test_data = scaler.transform(test_df)
    
    X_test = create_sequences(scaled_test_data, TIME_STEPS)

    y_test = test_labels[TIME_STEPS-1:]
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    return X_train, X_test, y_test, scaler

def create_sequences(data, time_steps):
    
    Xs = []
    for i in range(len(data) - time_steps + 1):
       
        Xs.append(data[i:(i + time_steps)])
    return np.array(Xs)

def calculate_reconstruction_error(sequences, predictions):
    
    return np.mean(np.abs(predictions - sequences), axis=1)
