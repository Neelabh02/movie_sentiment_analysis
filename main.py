import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

# mapping of words index label to word
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#load the pre trained model with relu function
model=load_model('simple_rnn_imdb.h5')


# helper function
#function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess user input
import re
def preprocess_text(text):

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

    words = text.split()

    print("Words:", words)

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    print("Encoded:", encoded_review)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review

# prediction function
def prediction_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]
    

#design streamlit app
import streamlit as st
st.title('IMDB Movie Review')    
st.write("enter the movie to classify it as positive or negative")

#user input
user_input=st.text_area("Movie Review")
if st.button("Classify"):
    if user_input.strip():
        sentiment, score = prediction_sentiment(user_input)

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.4f}")
    else:
        st.warning("Please enter a movie review.")
