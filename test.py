import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

# New labels
labels = ["A", "are", "B", "C", "Good", "Hi", "How", "I", "I Love You", "M", "No", "Peace","Thank", "Welcome", "Who", "Yes", "You", "Please", "Bad", "Sad", "Angry", "Help", "See", "Later", "Water", "Look", "Bathroom", "Sorry", "Happy", "Awesome"]

while True:
    success, img = cap.read()

    # Flip the frame horizontally
    img = cv2.flip(img, 1)

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if the cropped hand region is not empty
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            # Perform resizing and other operations here
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

            # Check if the index is within the valid range of the labels list
            if 0 <= index < len(labels):
                print(labels[index])

                # Calculate the width of the text
                text_width, _ = cv2.getTextSize(labels[index], cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)[0]

                # Extend the purple background based on the text width
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + text_width + 20, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)

                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            else:
                print("Error: Index out of range")

        else:
            print("Error: Empty hand region detected")

    cv2.imshow("Image", imgOutput)
    # Check if 'q' is pressed to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
