import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Przygotowanie danych
df = pd.read_csv('C:\\ai_python\\pl_emotions_from_text.csv')
df = df.dropna()
df['text'] = df['text'].astype(str)

X = df['text']
y = df['emotion']

le = LabelEncoder()
y = le.fit_transform(y)

# Tokenizacja i padding
max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

max_length = 50
X = pad_sequences(X, maxlen=max_length, truncating='post', padding='post')

# Podział danych
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def build_model(hp):
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    
    model = Sequential()
    
    # Embedding
    embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=256, step=32)
    model.add(Embedding(vocab_size, embedding_dim))
    
    # LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        lstm_units = hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32)
        return_sequences = i < hp.Int('num_lstm_layers', 1, 3) - 1
        
        if hp.Boolean('bidirectional'):
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=return_sequences)))
        else:
            model.add(LSTM(lstm_units, return_sequences=return_sequences))
            
        model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    
    # Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'dense_units_{i}', min_value=16, max_value=128, step=16),
            activation='relu'
        ))
        model.add(Dropout(hp.Float(f'dense_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(len(np.unique(y)), activation='softmax'))
    
    # Kompilacja
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Utworzenie katalogu dla logów TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# Inicjalizacja BayesianOptimization
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=30,  # liczba różnych konfiguracji do przetestowania
    num_initial_points=3,  # liczba początkowych losowych prób
    directory='keras_tuner',
    project_name='emotion_classification_bayesian'
)

# Callbacks dla treningu
stop_early = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Wyświetlenie podsumowania wyszukiwania
tuner.search_space_summary()

# Wyszukiwanie najlepszych hiperparametrów
tuner.search(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[stop_early, tensorboard_callback]
)

# Pobranie najlepszego modelu i wyświetlenie wyników
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nNajlepsze hiperparametry:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Trenowanie najlepszego modelu
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[stop_early, tensorboard_callback]
)

# Ewaluacja najlepszego modelu
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Wykresy
plt.figure(figsize=(12, 4))

# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Wykres funkcji straty
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predykcje i macierz pomyłek
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Szczegółowy raport klasyfikacji
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred_classes)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Zapisanie najlepszego modelu
model_save_path = "best_emotions_model_bayesian.h5" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
best_model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Zapisanie najlepszego modelu
model_save_path = "best_emotions_model_bayesian.h5"
best_model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Zapisanie tokenizera
import pickle
tokenizer_save_path = "tokenizer.pickle"
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved to: {tokenizer_save_path}")

# Zapisanie label encodera
label_encoder_save_path = "label_encoder.pickle"
with open(label_encoder_save_path, 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Label encoder saved to: {label_encoder_save_path}")

# Wyświetlenie informacji o uruchomieniu TensorBoarda
print("\nAby uruchomić TensorBoard, wykonaj w terminalu komendę:")
print(f"tensorboard --logdir {log_dir}")