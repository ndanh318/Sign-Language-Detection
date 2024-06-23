import cv2
import HandTracking
import numpy as np
import math
import time
import os

data_path = 'C:\Code\ComputerVision\SignLanguageDetection\V3\Data\Data_Number_05'
# labels = ['A', 'B', 'C', 'D', 'DD', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Mu', 'Munguoc', 'Rau']
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

count = 0
imgCount = 300
imgSize = 300
current_alphabet = '7'

# Create data folders
for label in labels:
    try:
        os.makedirs(os.path.join(data_path, label))
    except:
        pass

# Camera
cap = cv2.VideoCapture(0)
detector = HandTracking.handDetector()

while True:
    success, image = cap.read()
    # Detector
    hands, image = detector.findHands(image, handType=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = image[y - 20:y + h + 20, x - 20:x + w + 20]
        
        aspecRatio = h/w
        if aspecRatio > 1:
            n = imgSize/h
            wCal = math.ceil(n*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            n = imgSize/w
            hCal = math.ceil(n*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
        cv2.imshow('Image Crop', imgCrop)
        cv2.imshow('Image White', imgWhite)

    # Display
    cv2.imshow('Data Collector', image)

    # # Delete all file saved
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     files = os.listdir(data_path)
    #     for file_name in files:
    #         file_path = os.path.join(data_path, file_name)
    #         try:
    #             os.remove(file_path)
    #             print(f"Deleted {file_path}")
    #         except Exception as e:
    #             print(f"Error to delete {file_path}: {e}")
    #     breaks

    # Save data by pressing 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        count += 1
        cv2.imwrite(f'{data_path}/{current_alphabet}/{time.time()}.jpg', imgWhite)
        print(f'Saving image {count}')
        if count == imgCount:
            break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()