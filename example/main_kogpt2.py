#py:3.10, tf:2.10
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 1000 # 모델마다 값이 다름

model_path = 'vdcnn_model_with_kogpt2.h5'
tokenizer_path = "tokenizer_with_kogpt2.pickle"

model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    
    return text

def predict_text(text):
    sentence = preprocess_text(text)
    encoded_sentence = tokenizer.encode_plus(sentence,
                                             max_length=maxlen,
                                             padding="max_length",
                                             truncation=True)['input_ids']
    sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
    prediction = model.predict(sentence_seq)[0][0]
    return prediction
    
while True:
    text = input("Enter the sentence you want to test: ")
    result = predict_text(text)
    if result >= 0.5:
        print("This sentence contains abusive language.")
    else:
        print("It's a normal sentence.")
