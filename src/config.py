import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'anomaly_autoencoder.h5')

TIME_STEPS = 30        
N_FEATURES = 29         
ENCODER_DIM = 128       
BOTTLENECK_DIM = 64     
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

ANOMALY_THRESHOLD_PERCENTILE = 95
