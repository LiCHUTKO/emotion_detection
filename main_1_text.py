import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths to the model, tokenizer, and label encoder
model_path = r"C:\Users\user\OneDrive\Pulpit\best_emotions_model_bayesian_20241115-031941.h5"
label_encoder_path = r"C:\ai_python\new_label_encoder.pickle"
tokenizer_path = r"C:\ai_python\new_tokenizer.pickle"

# Load the model, tokenizer, and label encoder
model = load_model(model_path)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-ząćęłńóśźż ]", "", text)
    return text

# Function to predict emotion
def predict_emotion(input_text):
    input_text = preprocess_text(input_text)
    tokens = tokenizer.texts_to_sequences([input_text])
    maxlen = model.input_shape[1]
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=maxlen)
    if not tokens[0]:
        return [("Error: Text not recognized", 0.0)]
    predictions = model.predict(padded_tokens)
    top_indices = np.argsort(predictions[0])[-2:][::-1]
    top_classes = label_encoder.inverse_transform(top_indices)
    top_scores = predictions[0][top_indices]
    return list(zip(top_classes, top_scores))

def train_model():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return

    try:
        # Load dataset
        data = pd.read_csv(file_path)

        # Ensure 'text' and 'emotion' columns exist
        if 'text' not in data.columns or 'emotion' not in data.columns:
            raise ValueError("CSV must contain 'text' and 'emotion' columns.")

        # Drop rows with missing values
        data = data.dropna(subset=['text', 'emotion'])

        # Convert all values in 'text' to strings
        data['text'] = data['text'].astype(str)

        # Preprocess text and labels
        texts = data['text'].map(preprocess_text).values
        labels = data['emotion'].values

        # Tokenize and encode
        sequences = tokenizer.texts_to_sequences(texts)
        max_word_index = model.layers[0].input_dim - 1  # Maximum valid index in the embedding layer
        sequences = [[token for token in seq if token <= max_word_index] for seq in sequences]

        # Pad sequences
        maxlen = model.input_shape[1]
        x_data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
        y_data = label_encoder.transform(labels)
        y_data = tf.keras.utils.to_categorical(y_data, num_classes=len(label_encoder.classes_))

        # Split data
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        # Retrain model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=32)

        # Save updated model
        model.save(model_path)

        # Calculate confusion matrix
        y_val_pred = model.predict(x_val)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true_classes = np.argmax(y_val, axis=1)
        cm = confusion_matrix(y_val_true_classes, y_val_pred_classes)

        # Display confusion matrix
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix - Validation Set")
        plt.show()

        messagebox.showinfo("Success", "Model retrained and confusion matrix displayed!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train the model: {e}")



# GUI using Tkinter
def classify_text():
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Error", "No text provided!")
        return
    result = predict_emotion(input_text)
    output_text = "\n".join([f"{cls}: {score:.2f}" for cls, score in result])
    messagebox.showinfo("Prediction", output_text)

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        text_input.delete("1.0", tk.END)
        text_input.insert(tk.END, content)

root = tk.Tk()
root.title("Emotion Classifier")

frame = tk.Frame(root)
frame.pack(pady=10, padx=10)

text_input = tk.Text(frame, height=10, width=50)
text_input.pack()

button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

btn_predict = tk.Button(button_frame, text="Classify Text", command=classify_text)
btn_predict.pack(side=tk.LEFT, padx=5)

btn_load = tk.Button(button_frame, text="Load File", command=load_file)
btn_load.pack(side=tk.LEFT, padx=5)

btn_train = tk.Button(button_frame, text="Train Model", command=train_model)
btn_train.pack(side=tk.LEFT, padx=5)

root.mainloop()
