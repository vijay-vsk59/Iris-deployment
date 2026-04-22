import streamlit as st
import pickle
import numpy as np
import sklearn

with open("model.pkl", "b") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Iris Flower Classification")
st.write("Enter the features below : ")

# Input fields
sl=st.number_input("Sepal Length : ",min=4.3,max=7.9)
sw=st.number_input("Sepal Length : ",min=2,max=4.4)
pl=st.number_input("Sepal Length : ",min=1,max=6.9)
pw=st.number_input("Sepal Length : ",min=0.1,max=2.5)

if st.button("Predict"):
     pr= model.predict([[sl,sw,pl,pw]])
    classes = ["Setosa", "Versicolor", "Virginica"]
     st.write(f"Prediction : {classes[pr[0]]}")
