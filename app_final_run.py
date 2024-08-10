import streamlit as st
import numpy as np
import pickle

# Load the model once
with open("university_lr.pkl", "rb") as f:
    clf = pickle.load(f)

def predict(data):
    return clf.predict(data)

st.title("Case Study On University Admission Prediction")
st.markdown("Let's Predict Admission Chances")

st.header("")
col1, col2 = st.columns(2)

with col1:
    G = st.sidebar.slider("GRE Score", 1.0, 340.0, 0.5)
    T = st.sidebar.slider("TOEFL Score", 1.0, 120.0, 0.5)
    U = st.sidebar.slider("University Rating", 1, 5, 1)
    S = st.sidebar.slider("SOP", 1, 5, 1)
    L = st.sidebar.slider("LOR", 1, 5, 1)
    C = st.sidebar.slider("CGPA", 1.0, 10.0, 0.5)
    R = st.sidebar.slider("Research", 0.0, 1.0, 0.5)

st.text('')
if st.button("Chance To Get Admission"):
    # Create a numpy array for the prediction
    input_data = np.array([[G, T, U, S, L, C, R]])
    # Get prediction
    result = predict(input_data)
    # Display result
    st.text(f"Admission Chance: {result[0]}")

st.markdown("Developed By Nishant Doma Sawaimoon at NIELT Daman.")
