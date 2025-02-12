import cv2
import numpy as np
import tensorflow as tf
import pygame
import time
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the emotion detection model
model_path = '/home/abhiram1289/emotion_model.h5'
model = load_model(model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Map emotions to music files
emotion_songs = {
    "Happy": "/home/abhiram1289/Music/[iSongs.info] 01 - Samajavaragamana.mp3",
    "Sad": "/home/abhiram1289/Music/[iSongs.info] 02 - Bheem For Ramaraju.mp3",
    "Angry": "/home/abhiram1289/Music/Jai Lava Kusa Naa Songs Ringtones.mp3",
    "Surprise": "/home/abhiram1289/Music/[iSongs.info] 06 - Theme Of Kalki.mp3",
    "Neutral": "/home/abhiram1289/Music/Jai Srinivasa Jai Venkatesa - Dvv Entertainments logo Song.mp3",
    "Fear": "/home/abhiram1289/Music/Hanuman Chalisa.mp3",
    "Disgust": "/home/abhiram1289/Music/[iSongs.info] 03 - Roar Of RRR.mp3"
}

# Initialize pygame mixer for playing music
pygame.mixer.init()

def play_music(emotion):
    """Plays the song corresponding to the detected emotion only if it's a new emotion."""
    if emotion in emotion_songs:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(emotion_songs[emotion])
        pygame.mixer.music.play(-1)

def stop_music():
    """Stops the music playback."""
    pygame.mixer.music.stop()

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.current_emotion = None
        self.last_capture_time = 0

    def transform(self, frame):
        if st.session_state.get("stop_stream", False):
            return frame.to_ndarray(format="bgr24")

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        detected_emotion = None

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.array(face, dtype=np.float32).reshape(1, 48, 48, 1) / 255.0

            current_time = time.time()
            if self.current_emotion is None or (current_time - self.last_capture_time >= 10):
                prediction = model.predict(face)
                emotion_index = np.argmax(prediction)
                detected_emotion = emotion_labels[emotion_index]
                self.last_capture_time = current_time

                if detected_emotion != self.current_emotion:
                    self.current_emotion = detected_emotion
                    play_music(self.current_emotion)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, self.current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return img

# Streamlit UI
st.title("Emotion-Based Music Player")
st.write("This application detects your emotion through the webcam and plays music accordingly. By start and stop buttons you can control the emotion display on screen.")

# Initialize session state for stopping the stream
if "stop_stream" not in st.session_state:
    st.session_state["stop_stream"] = False

webrtc_ctx = webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)

if st.button("Stop Music"):
    stop_music()
    st.session_state["stop_stream"] = True  # Set flag to stop processing
    st.rerun()  # Refresh Streamlit UI

