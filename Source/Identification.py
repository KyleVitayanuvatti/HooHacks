import numpy as np
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from Text2Speech import speak, audio_thread

cap = cv2.VideoCapture(0)
sensor = HandDetector(maxHands=1)
classifier = Classifier("AI Model/image_recognition_model.h5", "AI Model/labels.txt")
imgSize = 224
offset = 40
labels = ['Hello', 'I', 'Am', 'Sunny', 'Glasses', 'Your', 'Communication', 'Assistant']

# Initialize imgWhite with a default value
imgWhite = np.zeros((imgSize, imgSize, 3), np.uint8)

# Initialize variables for tracking sign changes
prev_sign = None
last_speech_time = 0
speech_delay = 2  # Delay in seconds

# Initialize variable to store the time of the last prediction
last_prediction_time = 0
prediction_delay = 2  # Delay in seconds

while True:
    # Captures image frame
    captured, img = cap.read()
    imgOutput = img.copy()

    if not captured:
        continue

    # Detects any hand in camera view
    hand, img = sensor.findHands(img, draw=True)

    if hand:
        # Creates a bounding box for the first hand
        x, y, w, h = hand[0]['bbox']

        # Expands the bounding box and adjust values to handle errors
        x, y = max(0, x - offset), max(0, y - offset)
        w, h = min(w + 2 * offset, img.shape[1] - x), min(h + 2 * offset, img.shape[0] - y)

        # Crops the image to around bounding box
        imgCrop = img[y:y+h, x:x+w]

        # Makes sure imgCrop is not empty or it will crash
        if imgCrop.size == 0:
            continue

        # Calculates the resizing need for the cropped image to be layered on top of the white background
        aspectRatio = h / w
        if aspectRatio > 1:
            h2 = imgSize
            w2 = int(imgSize / aspectRatio)
        else:
            w2 = imgSize
            h2 = int(imgSize * aspectRatio)
        xOffset = (imgSize - w2) // 2
        yOffset = (imgSize - h2) // 2

        # Resizes the cropped image and places it on the top of the white background
        imgResize = cv2.resize(imgCrop, (w2, h2))
        imgWhite[yOffset:yOffset+h2, xOffset:xOffset+w2] = imgResize

        # Check if enough time has passed since the last prediction
        current_time = time.time()
        if current_time - last_prediction_time > prediction_delay:
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
            print(prediction, index)

            # Get the label text and calculate its width
            label_text = labels[index]
            (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)

            # Draw label background rectangle with adjusted width
            label_bg_color = (0, 120, 255)  # Blue color for better visibility of white text
            label_bg_x1 = x - offset
            label_bg_y1 = y - offset - 50
            label_bg_x2 = label_bg_x1 + label_width + 20
            label_bg_y2 = label_bg_y1 + label_height + 10
            cv2.rectangle(imgOutput, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), label_bg_color, cv2.FILLED)

            # Draw label text
            label_text_pos = (label_bg_x1 + 10, label_bg_y2 - 10)
            cv2.putText(imgOutput, label_text, label_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

            # Draw bounding box rectangle
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), label_bg_color, 4)

            # Check if the sign has changed and enough time has passed
            current_sign = labels[index]
            if current_sign != prev_sign and current_time - last_speech_time > speech_delay:
                audio_thread(current_sign)
                prev_sign = current_sign
                last_speech_time = current_time

            # Update the last prediction time
            last_prediction_time = current_time

    else:
        # Reset prev_sign when no hand is detected
        prev_sign = None

    # Displays the live feed for the cropped image on the top of the white background
    cv2.imshow('ImageWhite', imgWhite)

    # Displays the original live feed
    cv2.imshow('Image', imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()