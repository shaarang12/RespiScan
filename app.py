import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

def predict_placement(cgpa, iq, profile_score):
    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))

    if result[0] == 1:
        return 'placed'
    else:
        return 'not placed'

def main():
    st.title("Placement Predictor")

    st.write("Enter the following details:")
    
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
    iq = st.number_input("IQ", min_value=0, max_value=200, step=1)
    profile_score = st.number_input("Profile Score", min_value=0, max_value=100, step=1)

    if st.button("Predict Placement"):
        result = predict_placement(cgpa, iq, profile_score)
        st.write(f"The predicted placement result is: {result}")

        # Based on the result, show a link to a specific HTML page
        if result == 'placed':
            st.markdown("<a href='placed.html' target='_blank'>Click here for placement details</a>", unsafe_allow_html=True)
        else:
            st.markdown("<a href='not_placed.html' target='_blank'>Click here for more options</a>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
