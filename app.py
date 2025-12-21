import streamlit as st  # ← This line was missing!
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import os
import google.generativeai as genai

st.set_page_config(page_title="SignBridge", layout="centered")
st.title("🤟 SignBridge – Real-Time ISL Agent")
st.markdown("**Fingerspell word → Full ISL Animation + Gemini Agent Reply**")

# Gemini Setup
genai.configure(api_key="AIzaSyB0oBqPHFD9YN_qUChzEBeh8ur4TSYQY6Q")
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
           'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y])
        return np.array(landmarks).reshape(1, 42)
    return None

# Word → GIF
WORD_GIF = {
    "HELLO": "namaste-hello.gif",
    "THANKYOU": "thankyou.gif",
    "MOTHER": "mother.gif",
    "EAT": "eat.gif",
    "WATER": "water.gif"
}

if "word" not in st.session_state:
    st.session_state.word = ""

st.success(f"Current Word: {st.session_state.word.upper()}")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Clear Word"):
        st.session_state.word = ""
        st.rerun()

with col2:
    agent_btn = st.button("🤖 Ask Agent", type="primary")

# Camera input (photo mode — stable)
img_buffer = st.camera_input("Take photo of hand sign (hold letter steady)")

if img_buffer is not None:
    image = Image.open(img_buffer)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    st.image(image, caption="Your sign", use_column_width=True)
    
    lm = get_landmarks(frame)
    if lm is not None:
        pred = model.predict(lm, verbose=0)[0]
        idx = np.argmax(pred)
        conf = pred[idx]
        
        if conf > 0.8 and idx < 26:
            letter = letters[idx]
            st.session_state.word += letter
            st.success(f"Detected letter: {letter} (confidence: {conf:.2f})")
            st.rerun()

    # Show GIF when word complete
    upper = st.session_state.word.upper()
    if upper in WORD_GIF:
        path = os.path.join("ISL_GIFs", WORD_GIF[upper])
        if os.path.exists(path):
            st.image(path, caption=f"ISL Sign for: {upper}", use_column_width=True)
            st.balloons()

# Gemini Agent Reply
if agent_btn and st.session_state.word:
    with st.spinner("Agent thinking..."):
        prompt = f"A deaf user signed: '{st.session_state.word}'. Give a short, helpful reply in English."
        response = gemini.generate_content(prompt)
        st.success("**Agent Reply:**")
        st.write(response.text)

st.caption("Agentathon 2025 • Photo Mode • Gemini Agent • Team GDG-337")