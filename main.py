import numpy as np
import matplotlib.pyplot as plt
from src.data_processor import load_and_preprocess_data, calculate_reconstruction_error
from src.model_builder import build_lstm_autoencoder
from src.config import EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, ANOMALY_THRESHOLD_PERCENTILE, MODEL_PATH
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support

def train_and_evaluate():

    X_train, X_test, y_test, _ = load_and_preprocess_data()

    model = build_lstm_autoencoder()

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint, early_stopping],
        shuffle=False # Maintain time sequence integrity
    )

    model.load_weights(MODEL_PATH)

    X_train_pred = model.predict(X_train)
    train_mae_loss = calculate_reconstruction_error(X_train, X_train_pred)

    threshold = np.percentile(train_mae_loss, ANOMALY_THRESHOLD_PERCENTILE)
    print(f"\nCalculated Anomaly Threshold (95th percentile of Train MAE): {threshold:.4f}")

    X_test_pred = model.predict(X_test)
    test_mae_loss = calculate_reconstruction_error(X_test, X_test_pred)

    y_pred = (test_mae_loss > threshold).astype(int)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print("\n--- Final Model Performance (on Test Set) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f} (Target Metric)")

    plt.figure(figsize=(10, 6))
    plt.hist(test_mae_loss[y_test == 0], bins=50, density=True, label='Normal Loss (0)', alpha=0.6)
    plt.hist(test_mae_loss[y_test == 1], bins=50, density=True, label='Anomaly Loss (1)', alpha=0.6)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Absolute Error (MAE) Loss')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
