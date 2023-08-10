import streamlit as st
import pickle
import numpy as np

# Load the models
LR_model = pickle.load(open('modelLR.pkl', 'rb'))
# DT_model = pickle.load(open('modelDT.pkl', 'rb'))
# KNN_model = pickle.load(open('modelKNN.pkl', 'rb'))
# GNB_model = pickle.load(open('modelGNB.pkl', 'rb'))
# MNB_model = pickle.load(open('modelMNB.pkl', 'rb'))
# SVC_model = pickle.load(open('modelSVC.pkl', 'rb'))
# RF_model = pickle.load(open('modelRF.pkl', 'rb'))
# MLP_model = pickle.load(open('modelMLP.pkl', 'rb'))
# LGB_model = pickle.load(open('modelGB.pkl', 'rb'))

def main():
    st.title("Health Prediction App")
    st.write("Please fill out the form below:")

    # Form inputs
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=1, value=18)
    smoking = st.radio("Smoking", ['No', 'Yes'])
    yellow_fingers = st.radio("Yellow Fingers", ['No', 'Yes'])
    anxiety = st.radio("Anxiety", ['No', 'Yes'])
    peer_pressure = st.radio("Peer Pressure", ['No', 'Yes'])
    chronic_disease = st.radio("Chronic Disease", ['No', 'Yes'])
    fatigue = st.radio("Fatigue", ['No', 'Yes'])
    allergy = st.radio("Allergy", ['No', 'Yes'])
    wheezing = st.radio("Wheezing", ['No', 'Yes'])
    alcohol = st.radio("Alcohol", ['No', 'Yes'])
    coughing = st.radio("Coughing", ['No', 'Yes'])
    shortness_of_breath = st.radio("Shortness of Breath", ['No', 'Yes'])
    swallowing_difficulty = st.radio("Swallowing Difficulty", ['No', 'Yes'])
    chest_pain = st.radio("Chest Pain", ['No', 'Yes'])

    # Calculate derived features
    anxyf = int(anxiety == 'Yes' and yellow_fingers == 'Yes')

    data = [[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
             allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain, anxyf]]

    # Make predictions
    models = [LR_model]
    #, DT_model, KNN_model, GNB_model, MNB_model, SVC_model, RF_model, MLP_model, LGB_model
    accuracy = [98.33, 93.33, 92.5, 95.83, 95.83, 65.83, 95.83, 100, 98.33]

    predictions = [model.predict(data)[0] for model in models]
    weighted_sum_positive = sum(accuracy[i] for i, pred in enumerate(predictions) if pred == 1)
    weighted_sum_negative = sum(accuracy[i] for i, pred in enumerate(predictions) if pred == 0)

    positive_percentage = int((weighted_sum_positive / len(models)) * 100)
    negative_percentage = int((weighted_sum_negative / len(models)) * 100)

    # Display result
    if positive_percentage > negative_percentage:
        st.write(f"Prediction: Health Risk - Positive ({positive_percentage}% confidence)")
        # Add more details or visualizations as needed for a positive prediction
    else:
        st.write(f"Prediction: Health Risk - Negative ({negative_percentage}% confidence)")
        # Add more details or visualizations as needed for a negative prediction

if __name__ == "__main__":
    main()
