%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

st.set_page_config(page_title="SignBridge Agent", layout="centered")
st.title("Global Real-Time Sign Language Translator Agent")
st.markdown("### Agentathon 2025 – Making the world accessible")

# Load your model (change "model.h5" to your actual model file name/path)
@st.cache_resource
def load_model():
    # If your model is not uploaded yet, comment this out for now
    # return tf.keras.models.load_model("model.h5")
    return None  # Dummy for testing – we'll add your real model next

model = load_model()

# Initialize MediaPipe Hands (for hand detection in signs)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Dummy prediction function (replace with your real SignTalk code later)
def predict_sign(frame_rgb):
    # Example: Detect hands and return a dummy sign
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        return "HELLO 👋 (Hands detected!)"
    return "NO SIGN DETECTED"

# UI Elements
frame_window = st.empty()
output_text = st.empty()
status = st.empty()

if st.button("Start Live Camera", type="primary"):
    status.success("Camera ON – Start signing!")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB")
        
        predicted = predict_sign(frame_rgb)
        output_text.markdown(f"**Translated →** {predicted}")
        
        # Stop on 'q' key (press in terminal if needed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    hands.close()
    status.info("Camera stopped – Success!")
