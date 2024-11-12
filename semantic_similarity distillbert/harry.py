import os
import numpy as np
import pandas as pd
import streamlit as st

st.title("🪄🪄🪄 Semantic Search on Harry Potter 🪄🪄🪄")
st.image("harry.jpg")
st.sidebar.write("Enter your wizarding document:")
file=st.sidebar.file_uploader("input your doc:")


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')


import pandas as pd
if file:
    harry = pd.read_excel(file,header=None)

    harry=harry.iloc[:,0].apply(lambda x: x.lstrip())
    k=st.sidebar.slider('Number of results',0,len(harry))
    def find_similarity(vector_rep, all_representations,k=1):
        similarities = cosine_similarity(vector_rep,all_representations)
        if k==1:
            return similarities.argmax()
        else:
            s=similarities[0]
            return s.argsort()[::-1][:k]

    user_input=st.sidebar.text_input("Enter your query:")

    embeddings = model.encode(harry.values)
    if user_input:
        distilbert_similar_indexes = find_similarity(model.encode([user_input]),embeddings,2)

        output_data = []
        for index in distilbert_similar_indexes:
            output_data.append(harry[index])

        with st.chat_message("Harry"):
            st.write(" 🧙🧙 Your answer by Tom Marvolo Riddle is: 🪄🪄🪄")
            for i in range(k):
                st.write(output_data[i])
                
    
        