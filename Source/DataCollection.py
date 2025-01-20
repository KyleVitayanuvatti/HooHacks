import numpy as np
import time
import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
sensor = HandDetector(maxHands=1)

#Change path according to the class
folder = "/Users/kylevitayanuvatti/Desktop/HooHacks/Data/Assistant"

imgSize = 224
offset = 40
counter = 0

while True:
    #Captures image frame 
    captured, img = cap.read()

    if not captured:
        continue

    #Detects any hand in camera view
    hand, img = sensor.findHands(img)

    if hand:
        #Creates a bounding box for the first hand 
        x, y, w, h = hand[0]['bbox']

        #Expands the bounding box and adjust values to handle errors
        x, y = max(0, x-offset), max(0, y-offset)
        w, h = min(w + 2*offset, img.shape[1] - x), min(h + 2*offset, img.shape[0] - y)

        #Makes a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        #Crops the image to around bounding box
        imgCrop = img[y:y+h, x:x+w]

        #Makes sure imgCrop is not empty or it will crash
        if imgCrop.size == 0:
            continue

        #Calculates the resizing need for the croped image to be layered on top of the white background
        aspectRatio = h / w
        if aspectRatio > 1:
            h2 = imgSize
            w2 = int(imgSize / aspectRatio)
        else:
            w2 = imgSize
            h2 = int(imgSize * aspectRatio)

        xOffset = (imgSize - w2) // 2
        yOffset = (imgSize - h2) // 2

        #Resizes the cropped image and places it on the top of the white background
        imgResize = cv2.resize(imgCrop, (w2, h2))
        imgWhite[yOffset:yOffset+h2, xOffset:xOffset+w2] = imgResize

        #Displays the live feed for the croped image on the top of the white background
        cv2.imshow('ImageWhite', imgWhite)

    #Displays the original live feed
    cv2.imshow('Image', img)

    #Saves the image when 'i' is pressed or held(used 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        counter =+ 1


