import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# Ścieżka do pliku z danymi
data_path = r"C:\ai_python\pl_emotions_from_text_2_0.csv"

# Wczytanie danych
data = pd.read_csv(data_path)

# Zakładam, że dane mają kolumny 'text' (tekst) i 'label' (etykieta emocji)
texts = data['text'].astype(str).tolist()  # Lista tekstów
labels = data['emotion'].tolist()            # Lista etykiet

# 1. Tworzenie tokenizera
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # oov_token obsługuje słowa spoza słownika
tokenizer.fit_on_texts(texts)

# Zapisanie tokenizera
tokenizer_path = r"C:\ai_python\new_tokenizer.pickle"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer zapisany do: {tokenizer_path}")

# 2. Tworzenie enkodera etykiet
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Zapisanie enkodera
encoder_path = r"C:\ai_python\new_label_encoder.pickle"
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Encoder zapisany do: {encoder_path}")

# Wyświetlenie informacji
print(f"Ilość słów w tokenizerze: {len(tokenizer.word_index)}")
print(f"Klasy emocji: {list(label_encoder.classes_)}")
