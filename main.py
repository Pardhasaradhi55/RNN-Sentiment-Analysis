import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

word_index=imdb.get_word_index()
reverse_word_index={value:key for key, value in word_index.items()}
from my_layers import MyCustomLayer
model = keras.saving.load_model('simple_rnn_imdb.keras', custom_objects={'MyCustomLayer': MyCustomLayer})

#model = keras.saving.load_model('simple_rnn_imdb.keras')
#model=load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


##prediction function
def predict_sentiment(review):
    processed_input=preprocess_text(review)

    prediction=model.predict(processed_input)

    sentiment='Positive' if prediction[0][0] >0.5 else 'Negetive'

    return sentiment,prediction[0][0]


import streamlit as st
st.title('IMDB Movie Review Analysis')
st.write('Leave your review about the movie')

#user input
user_input=st.text_area('MOVIE REVIEW')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] >0.5 else 'Negetive'

    #display result
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction score : {prediction[0][0]}')

else:
    st.write('please leave a review')
