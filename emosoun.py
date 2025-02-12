import cv2
import numpy as np
import tensorflow as tf
import pygame
from keras.models import load_model
import time

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
        pygame.mixer.music.stop()  # Stop current song
        pygame.mixer.music.load(emotion_songs[emotion])
        pygame.mixer.music.play(-1)  # Loop indefinitely

def detect_emotion():
    """Detects emotion from the webcam, updates it on screen in real-time, and plays music only when a new emotion is detected."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    current_emotion = None  # Store last detected emotion
    last_capture_time = 0  # Time tracking for 10-sec interval

    while True:  # Continuous Loop
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if the frame is not captured

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = None  # Default to None if no face detected

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.array(face, dtype=np.float32).reshape(1, 48, 48, 1) / 255.0  # Normalize
            
            # Capture emotion immediately at the start and then every 10 seconds
            current_time = time.time()
            if current_emotion is None or (current_time - last_capture_time >= 10):  
                prediction = model.predict(face)
                emotion_index = np.argmax(prediction)
                detected_emotion = emotion_labels[emotion_index]
                last_capture_time = current_time  # Update last capture time

                # If a new emotion is detected, update the music
                if detected_emotion != current_emotion:
                    current_emotion = detected_emotion
                    play_music(current_emotion)

            # Draw face box and detected emotion on screen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show webcam feed with detected emotion
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

if __name__ == '__main__':
    detect_emotion()

