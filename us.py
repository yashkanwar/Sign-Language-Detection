import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import streamlit as st

# Set up video capture and hand detection
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Increase maxHands to 2 for detecting multiple hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# Labels for words and phrases
labels = ["A", "are", "B", "C", "Good", "Hi", "How", "I", "I Love You", "M", "No", "Peace","Thank", "Welcome", "Who", "Yes", "You", "Please", "Bad", "Sad", "Angry", "Help", "See", "Later", "Water", "Look", "Bathroom", "Sorry", "Happy", "Awesome" ]

# Display the UI
st.title("HandSign Language Detection:")

# Create a placeholder for the video feed
video_placeholder = st.empty()


# Function to capture and process video frames
def capture_frames():
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        imgOutput = img.copy()
        hands, _ = detector.findHands(img)
        if hands:
            for hand in hands:  # Iterate over all detected hands
                x, y, w, h = hand['bbox']
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    aspectRatio = h / w
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    if 0 <= index < len(labels):
                        # Calculate the width of the text
                        text_width, _ = cv2.getTextSize(labels[index], cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)[0]

                        # Extend the purple background based on the text width
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                      (x - offset + text_width + 20, y - offset - 50 + 50), (255, 0, 255),
                                      cv2.FILLED)

                        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                    (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset),
                                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Display the video feed with predictions
        video_placeholder.image(imgOutput, channels="BGR", use_column_width=True)


# Call the capture_frames function
capture_frames()
