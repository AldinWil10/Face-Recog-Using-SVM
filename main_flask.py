from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

app = Flask(__name__)

def generate_frames():
    
    cap = cv.VideoCapture(0)
    #0 untuk HIK
    #1 untuk REXUS

    cap.set(3, 1280) #640, 1280, 1920
    cap.set(4, 720) #480, 720, 1080
    
    #INITIALIZE
    facenet = FaceNet()
    faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)
    haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = pickle.load(open("svm_model_160x160.pkl", 'rb'))
    while cap.isOpened():
        _, frame = cap.read()
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
            
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3, cv.LINE_AA)
            
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    cap.release()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
