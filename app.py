import streamlit as st
import pickle
import numpy as np

# Load the models
LR_model = pickle.load(open('modelLR.pkl', 'rb'))

def main():
    st.title("Health Prediction App")
    st.write("Please fill out the form below:")

    # Form inputs
    input_mapping = {
        'Male': 0, 'Female': 1,
        'No': 0, 'Yes': 1
    }

    gender = st.radio("Gender", ['Male', 'Female'], format_func=lambda x: input_mapping[x])
    age = st.number_input("Age", min_value=1, value=18)
    smoking = st.radio("Smoking", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    yellow_fingers = st.radio("Yellow Fingers", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    anxiety = st.radio("Anxiety", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    peer_pressure = st.radio("Peer Pressure", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    chronic_disease = st.radio("Chronic Disease", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    fatigue = st.radio("Fatigue", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    allergy = st.radio("Allergy", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    wheezing = st.radio("Wheezing", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    alcohol = st.radio("Alcohol", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    coughing = st.radio("Coughing", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    shortness_of_breath = st.radio("Shortness of Breath", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    swallowing_difficulty = st.radio("Swallowing Difficulty", ['No', 'Yes'], format_func=lambda x: input_mapping[x])
    chest_pain = st.radio("Chest Pain", ['No', 'Yes'], format_func=lambda x: input_mapping[x])

    # Calculate derived features
    anxyf = int(anxiety == 1 and yellow_fingers == 1)

    data = [[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
             allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain, anxyf]]

    # Make predictions
    models = [LR_model]
    accuracy = [98.33]

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
