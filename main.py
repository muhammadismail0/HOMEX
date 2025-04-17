import pandas as pd
import torch
import streamlit as st
import pyttsx3
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
import threading

# Load model and dataset
@st.cache_resource
def load_model():
    return SentenceTransformer("model/property-similarity-model")

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Abubakar/sementic/dataset/dataset.csv")
    df["combined_property"] = df["property_1"].fillna("") + " " + df["property_2"].fillna("")
    return df

model = load_model()
df = load_data()

# Global variables
speech_thread = None
stop_flag = threading.Event()

# Text-to-Speech functions
def speak_text(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        try:
            engine.runAndWait()
        except RuntimeError:
            pass

    global speech_thread
    stop_flag.clear()

    if speech_thread and speech_thread.is_alive():
        stop_flag.set()
        speech_thread.join()

    speech_thread = threading.Thread(target=run)
    speech_thread.start()

# Stop Speaking
def stop_speaking():
    stop_flag.set()
    engine = pyttsx3.init()
    engine.stop()

# Voice Input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand.")
    except sr.RequestError:
        st.error("⚠️ Internet issue.")
    return ""

# UI
st.title("🏡 Property Semantic Search")
input_method = st.radio("Choose Input Method:", ("📝 Text", "🎙️ Voice"))

user_input = ""
search_triggered = False

if input_method == "📝 Text":
    user_input = st.text_input("🔍 Enter your requirement:")
    search_triggered = st.button("🔎 Search")
else:
    if st.button("🎧 Listen"):
        user_input = get_voice_input()
        search_triggered = True

# Stop audio button
if st.button("🛑 Stop Speaking"):
    stop_speaking()

# Perform search
if user_input and search_triggered:
    property_embeddings = model.encode(df["combined_property"].tolist(), convert_to_tensor=True)
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, property_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=5)

    st.subheader("🏆 Top 5 Matching Properties:")
    for score, idx in zip(top_results[0], top_results[1]):
        result_text = df.iloc[int(idx)]['combined_property']
        st.markdown(f"**{result_text}**")
        st.write(f"🧠 Similarity Score: {score.item():.4f}")
        st.markdown("---")
        speak_text(f"Match found: {result_text}")
