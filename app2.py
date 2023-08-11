import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('path_to_your_model')

# Define class labels
class_labels = []  # Replace with your actual class labels

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title("Image Classification App")
    st.write("Upload an image for prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            # Preprocess the image
            input_image = preprocess_image(image)

            # Perform prediction
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions[0])

            st.write(f"Predicted class: {class_labels[predicted_class]}")

if __name__ == "__main__":
    main()
