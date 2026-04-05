# 🎭 Emotion Recognition App (Streamlit + Deep Learning)

This project detects human facial emotions from an uploaded image using a Convolutional Neural Network (CNN).  
The model was trained on the FER2013 dataset and deployed using **Streamlit**.

---

## ⭐ Features
- Upload an image (JPG, JPEG, PNG)
- Automatically detect faces using Haar Cascade
- Crop and process the detected face
- Predict emotion using a trained deep learning model
- Displays the emotion result clearly

---

## 🧠 Emotions Detected
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## 📁 Project Structure
FaceRecognitionExpression/
│
├── emotion.py
├── emotion_model.h5
├── emotion_labels.json
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md

## 🚀 How to Run the App

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ streamlit run emotion.py


Requirements

These dependencies are needed to run the app:
streamlit
tensorflow
opencv-python-headless
numpy
Pillow

📸 Sample Usage
1.Upload an image containing a face
2.The app will detect the face
3.The face is resized to 48×48
4.The model predicts an emotion
5.The predicted emotion is displayed

📦 Model Files
emotion_model.h5 → Your trained CNN model
emotion_labels.json → Mapping from number → emotion name

Place these files in the same folder as emotion.py.

## 🔗 GitHub Link

Check out the full project here: [ArinolaDev GitHub](https://github.com/ArinolaDev)

👨‍💻 Author
Arinola
Email: arinolasmart@gmail.com
Feel free to reach me for any questions or improvements!

