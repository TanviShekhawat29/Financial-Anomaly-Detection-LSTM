from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from src.config import TIME_STEPS, N_FEATURES, ENCODER_DIM, BOTTLENECK_DIM, LEARNING_RATE

def build_lstm_autoencoder():

    inputs = Input(shape=(TIME_STEPS, N_FEATURES))

    L1 = LSTM(ENCODER_DIM, activation='relu', return_sequences=True)(inputs)

    L2 = LSTM(BOTTLENECK_DIM, activation='relu', return_sequences=False)(L1)

    L3 = RepeatVector(TIME_STEPS)(L2)

    L4 = LSTM(BOTTLENECK_DIM, activation='relu', return_sequences=True)(L3)

    L5 = LSTM(ENCODER_DIM, activation='relu', return_sequences=True)(L4)

    output = TimeDistributed(Dense(N_FEATURES))(L5)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mae')
    
    print(model.summary())
    return model
