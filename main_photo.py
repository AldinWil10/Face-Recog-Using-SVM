#IMPORT
import cv2 as cv
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Load images from folder
image_folder = "dataset/Will"
image_files = os.listdir(image_folder)

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv.imread(image_path)
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    
    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160)) # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        distance = model.decision_function(ypred)
        min_distance = np.min(distance)
        
        if min_distance > 0.16:
            final_name = "unknown"
        else:
            face_name = model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)
        
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)  # Ubah warna bingkai menjadi hijau
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 3, cv.LINE_AA)  # Ubah warna nama menjadi hijau

    cv.imshow("Face Recognition:", frame)
    cv.waitKey(0)  # Wait for a key press after displaying each image

cv.destroyAllWindows()
