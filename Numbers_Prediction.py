import cv2
import HandTracking
import numpy as np
import math
import os
from collections import Counter
from keras.utils import img_to_array, load_img
from keras.models import load_model
from Classification import Classifier

imgSize = 300
prediction = []
sentence = []


# Labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Model
model_path = "C:/Code/ComputerVision/SignLanguageDetection/V3/Model//NSLD_20240430_012510_Loss_0.047_Accuracy_0.966.h5"

def most_common_value(sequence):
    counter = Counter(sequence)
    most_common = counter.most_common(1)
    
    return most_common[0][0] if most_common else None


def main():
    cap = cv2.VideoCapture(0)
    detector = HandTracking.handDetector()
    classifier = Classifier(model_path)
    
    while cap.isOpened():
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
                
                results, index = classifier.prediction(imgWhite)
                prediction.append(np.argmax(results))
                if most_common_value(prediction[-25:]) == np.argmax(results):
                    number = labels[index]
                    sentence.append(number) if not sentence or sentence[-1] != number else None
                    print(number)
                
            else:
                n = imgSize/w
                hCal = math.ceil(n*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal)/2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                
                results, index = classifier.prediction(imgWhite)
                prediction.append(np.argmax(results))
                if most_common_value(prediction[-25:]) == np.argmax(results):
                    number = labels[index]
                    sentence.append(number) if not sentence or sentence[-1] != number else None
                    print(number)

            # Visualize
            cv2.rectangle(image, (x - 20, y - 20 - 50), (x - 20 + 90, y - 20 - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(image, number, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(image, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 255), 4)

        # cv2.imshow('Image Crop', imgCrop)
        # cv2.imshow('Image White', imgWhite)
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    