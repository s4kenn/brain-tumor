import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import tensorflow as tf
from PIL import Image
import numpy as np
import wikipediaapi
import pyttsx3
import pydicom
from lime import lime_image
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Brain Tumor Detection & Help Assist.",
    page_icon=":brain:",
    layout="wide",
)

# Google Gemini API credentials
api_key = os.getenv('AIzaSyChnJgAjjsqy2HFVfzcbsUBtpoIdTdls-s')  # Ensure this key is in your .env file
gen_ai.configure(api_key=api_key)
model_gemini = gen_ai.GenerativeModel('gemini-pro')

# Define categories and symptoms
categories = ["glioma", "meningioma", "no tumor", "pituitary"]

symptoms = {
    "Glioma": [
        "Headaches",
        "Seizures",
        "Vision problems",
        "Nausea and vomiting",
        "Difficulty with balance and coordination",
        "Changes in personality or behavior",
        "Weakness or numbness in limbs"
    ],
    "Meningioma": [
        "Headaches",
        "Seizures",
        "Vision changes",
        "Hearing loss or ringing in the ears",
        "Nausea and vomiting",
        "Weakness or numbness",
        "Difficulty with speech or movement"
    ],
    "No Tumor": [
        "No specific symptoms related to a tumor; symptoms are typically related to other conditions."
    ],
    "Pituitary Tumor": [
        "Headaches",
        "Vision problems",
        "Unexplained weight gain or loss",
        "Changes in menstrual cycle or sexual function",
        "Growth of hands and feet (in case of acromegaly)",
        "Fatigue",
        "Changes in mood or cognitive functions"
    ]
}

# Load the trained model with caching
@st.cache_data(show_spinner=False)
def load_model(version="v1"):
    model_path = 'brain_tumor_detection_model.h5'
    return tf.keras.models.load_model(model_path)

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Enhanced validation function using pydicom
def is_valid_mri(file):
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

# Function to predict the tumor type and confidence score
def predict(image):
    preprocessed_image = preprocess_image(image)
    try:
        predictions = model.predict(preprocessed_image)
        confidence = np.max(predictions) * 100  # Convert to percentage
        tumor_type = categories[np.argmax(predictions)]
        return tumor_type, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Function to add model explainability using LIME
def explain_prediction(image, model):
    try:
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to the required size
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

# Function to get tumor information from Wikipedia
def get_tumor_info(tumor_type):
    wiki_wiki = wikipediaapi.Wikipedia('en')  # Initialize the Wikipedia API
    try:
        page = wiki_wiki.page(tumor_type)
        if not page.exists():
            return f"No information found on Wikipedia for {tumor_type}."
        summary = page.summary[:500]  # Fetch a summary (adjust length as needed)
        return summary
    except wikipediaapi.exceptions.DisambiguationError:
        return f"Multiple results found for {tumor_type}. Please specify further."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to get a response from Google Gemini
def get_chatbot_response(query):
    try:
        response = model_gemini.generate_text(prompt=query)
        return response['text']
    except Exception as e:
        st.error(f"Error with Google Gemini API: {e}")
        return "Sorry, I couldn't fetch the response."

# Initialize TTS engine
# Removed pyttsx3 initialization to avoid local system dependency issues

# Function to speak out the response (if using cloud-based TTS)
# def speak_text(text):
#     try:
#         # Implement cloud-based TTS here
#         pass
#     except Exception as e:
#         st.error(f"Error with TTS: {e}")

# Translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

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
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                with st.spinner('Predicting...'):
                    prediction, confidence = predict(image)
                if prediction:
                    st.write(f"Prediction: {prediction}")
                    st.write(f"Confidence: {confidence:.2f}%")  # Display confidence score
                    explain_prediction(image, model)
                    tumor_info = get_tumor_info(prediction)
                    st.write(f"Tumor Information: {tumor_info}")
                    # speak_text(tumor_info)  # Commented out TTS function
        else:
            st.error("Invalid MRI image. Please upload a valid MRI image.")

# elif app_mode == "HELP ASSIST":
#     st.header("Chat with the Assist")

#     # Initialize chat session if not already present
#     if "chat_session" not in st.session_state:
#         st.session_state.chat_session = model_gemini.generate_text(prompt="Initialize chat session")

#     # Display the chat history
#     if "chat_session" in st.session_state:
#         st.write(st.session_state.chat_session)  # Modify as needed for proper chat history display

#     # Input field for user's message
#     user_prompt = st.chat_input("ASK FOR PRECAUTIONS:")
#     if user_prompt:
#         st.chat_message("user").markdown(user_prompt)
#         gemini_response = get_chatbot_response(user_prompt)
#         st.chat_message("assistant").markdown(gemini_response)
