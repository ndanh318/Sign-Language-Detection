import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

class Classifier:
    
    def __init__(self, model_path, labels_path=None):
        self.model_path = model_path
        self.labels_path = labels_path
        np.set_printoptions(suppress=True)  # Disable scientific notation for clarity
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Load model
        self.model = load_model(self.model_path)
        # Load labels (if labels file is provided)
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = [line.strip() for line in label_file]
            label_file.close()
        else:
            print("No Labels Found")
        
    def prediction(self, image, draw=True):
        # Preprocessing
        imgResize = cv2.resize(image, (224, 224))
        image_arr = np.asarray(imgResize)
        normalized_image_array = (image_arr.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        # Predict
        results = self.model.predict(self.data)
        prediction = np.argmax(results)
        
        if draw and self.labels_path:
            cv2.putText(image, str(self.list_labels[prediction[0]]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    
        return list(results[0]), prediction