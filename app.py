import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import tensorflow as tf
from PIL import Image
import numpy as np
import pydicom
from lime import lime_image
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Brain Tumor Detection & Assist",
    page_icon=":brain:",
    layout="wide",
)

# Google Gemini API credentials
api_key = os.getenv('AIzaSyChnJgAjjsqy2HFVfzcbsUBtpoIdTdls-s')  # Ensure this is stored in your .env file
gen_ai.configure(api_key=api_key)
model_gemini = gen_ai.GenerativeModel('gemini-pro')

# Define categories and symptoms
categories = ["glioma", "meningioma", "no tumor", "pituitary"]

symptoms = {
    "Glioma": ["Headaches", "Seizures", "Vision problems", "Nausea", "Difficulty with balance", "Personality changes", "Weakness in limbs"],
    "Meningioma": ["Headaches", "Seizures", "Vision changes", "Hearing loss", "Nausea", "Weakness", "Speech or movement difficulty"],
    "No Tumor": ["No tumor-specific symptoms."],
    "Pituitary Tumor": ["Headaches", "Vision problems", "Weight changes", "Mood swings", "Fatigue"],
}

# Load the trained model
def load_model():
    model_path = 'brain_tumor_detection_model.h5'
    return tf.keras.models.load_model(model_path)

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    """Preprocess the image to match model input."""
    # Ensure the image is in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to match the model's input size (assuming the model was trained on 150x150 images)
    img = image.resize((150, 150))
    
    # Normalize the image (as the model might have been trained with normalized inputs)
    img_array = np.array(img) / 255.0
    
    # Expand dimensions to make it compatible with the model input
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to resize the image for display
def resize_image_for_display(image, max_width=512):
    """Resize the image while maintaining aspect ratio for better display quality."""
    width, height = image.size
    if width > max_width:
        aspect_ratio = height / width
        new_height = int(aspect_ratio * max_width)
        image = image.resize((max_width, new_height), Image.ANTIALIAS)
    return image

# Enhanced validation function using pydicom
def is_valid_mri(file):
    """Check if the uploaded file is a valid MRI image."""
    try:
        dicom_file = pydicom.dcmread(file, force=True)
        if dicom_file.Modality == 'MR':
            return True
    except Exception as e:
        try:
            image = Image.open(file)
            return True
        except Exception as e:
            st.warning(f"Image validation failed: {e}")
    return False

# Function to predict the tumor type and confidence score with thresholding
def predict(image, threshold=75):
    """Predict tumor type and apply confidence threshold."""
    preprocessed_image = preprocess_image(image)
    try:
        # Get model predictions
        predictions = model.predict(preprocessed_image)
        
        # Print raw predictions for debugging
        st.write(f"Raw Predictions: {predictions}")
        
        # Get the highest confidence score and corresponding tumor type
        confidence = np.max(predictions) * 100  # Convert to percentage
        tumor_type = categories[np.argmax(predictions)]
        
        # If confidence is below a threshold, raise uncertainty
        if confidence < threshold:
            tumor_type = "Potential tumor, please verify with a doctor"
        
        return tumor_type, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Function to add model explainability using LIME
def explain_prediction(image, model):
    """Use LIME to explain the model's prediction."""
    try:
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to the required size for LIME
        img_array = np.array(image.resize((150, 150)))
        
        # Initialize LIME Image Explainer
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array,
            model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )
        
        st.subheader("OUTPUT of the SCAN :")
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        plt.imshow(temp)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in explanation: {e}")

# Streamlit app interface
st.title("Brain Tumor Detection & ChatBot")
st.markdown("### Disclaimer: I am only for informational purposes and should not replace a doctor's diagnosis.")
st.markdown("Don't depend on me completely")

# Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select a mode", ["Diagnosis Test", "Tumor Detection"])

if app_mode == "Diagnosis Test":
    st.header("Diagnosis Test")
    st.write("Please select the symptoms you are experiencing.")

    symptom_selection = {}
    for symptom_list in symptoms.values():
        for symptom in symptom_list:
            if symptom not in symptom_selection:
                symptom_selection[symptom] = st.checkbox(symptom, key=symptom)
    
    if st.button("Diagnose"):
        selected_symptoms = [symptom for symptom, selected in symptom_selection.items() if selected]
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            st.write(f"Selected Symptoms: {', '.join(selected_symptoms)}")
            probabilities = {condition: 0 for condition in symptoms.keys()}

            for condition, symptom_list in symptoms.items():
                match_count = sum(1 for symptom in selected_symptoms if symptom in symptom_list)
                probability = (match_count / len(symptom_list)) * 100
                probabilities[condition] = probability

            max_prob = max(probabilities.values())
            st.write("Diagnosis Result:")
            for condition, probability in probabilities.items():
                st.write(f"{condition}: {probability:.2f}%")

            if list(probabilities.values()).count(max_prob) > 1:
                st.warning("There may be a clash between multiple conditions. Please consult a doctor.")
            elif max_prob == 0:
                st.warning("No matching conditions found. Please consult a doctor for further evaluation.")

elif app_mode == "Tumor Detection":
    st.header("Upload MRI for Tumor Detection")
    uploaded_file = st.file_uploader("Upload an MRI image", type=["dcm", "jpg", "png", "jpeg"])

    if uploaded_file is not None:
        if is_valid_mri(uploaded_file):
            image = Image.open(uploaded_file)
            resized_image = resize_image_for_display(image)  # Resize the image for better display
            st.image(resized_image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                with st.spinner('Predicting....'):
                    prediction, confidence = predict(image)
                if prediction:
                    st.write(f"Prediction: {prediction}")
                    st.write(f"Confidence: {confidence:.2f}%")  # Display confidence score
                    explain_prediction(image, model)
        else:
            st.error("Invalid MRI image. Please upload a valid MRI image.")
