import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("emotion_model.h5")

# Load emotion labels
with open("emotion_labels.json", "r") as f:
    emotion_dict = json.load(f)

# Load Haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("Face Emotion Recognition App")
st.write("Upload an image and the AI will detect the emotion!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("❌ No face detected in the image.")
    else:
        (x, y, w, h) = faces[0]

    
        face = gray[y:y+h, x:x+w]

        
        resized = cv2.resize(face, (48, 48))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)

    
        preds = model.predict(reshaped)
        emo_idx = int(np.argmax(preds))
        emotion = emotion_dict[str(emo_idx)]

        st.subheader("🎯 Prediction:")
        st.success(f"Detected Emotion: **{emotion}**")

        st.image(resized, caption="Cropped Face (Used for Prediction)", width=200)

st.subheader("Use Webcam for Emotion Detection")

camera_image = st.camera_input("Take a picture with your webcam")

if camera_image is not None:
    image = Image.open(camera_image)
    img_np = np.array(image)

    # FIX 1: Convert Streamlit RGB → OpenCV BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # (Optional) Light enhancement
    img_np = cv2.convertScaleAbs(img_np, alpha=1.2, beta=20)

    st.image(image, caption="Webcam Capture", use_column_width=True)

    # Face detection MUST use BGR → GRAY
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # FIX 2: Make face detection more sensitive
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        st.error("❌ No face detected in webcam image.")
    else:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        resized = cv2.resize(face, (48, 48))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)

        preds = model.predict(reshaped)
        emo_idx = int(np.argmax(preds))
        emotion = emotion_dict[str(emo_idx)]

        st.subheader("Prediction (Webcam):")
        st.success(f"Detected Emotion: {emotion}")

        st.image(resized, caption="Cropped Face (Used for Prediction)", width=200)
    
