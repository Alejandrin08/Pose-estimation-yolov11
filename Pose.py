import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ultralytics import YOLO
from math import sqrt
from deepface import DeepFace

model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

data_list = []

left_wrist_x_1, left_wrist_y_1 = [], []
right_wrist_x_1, right_wrist_y_1 = [], []
left_wrist_x_2, left_wrist_y_2 = [], []
right_wrist_x_2, right_wrist_y_2 = [], []

json_file = os.path.join(os.path.dirname(__file__), "pose_data.json")

def calculate_distance(point1, point2):
    return sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

fig, ax = plt.subplots()
left_line_1, = ax.plot([], [], 'ro-', label="Muñeca izq Persona 1")
right_line_1, = ax.plot([], [], 'bo-', label="Muñeca der Persona 1")
left_line_2, = ax.plot([], [], 'go-', label="Muñeca izq Persona 2")
right_line_2, = ax.plot([], [], 'mo-', label="Muñeca der Persona 2")
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
ax.invert_yaxis()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

def detect_emotion(frame, face):
    x, y, w, h = face
    face_crop = frame[y:y+h, x:x+w]
    try:
        analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except:
        return "No detectado"

def graphic(frame):
    global right_wrist_x_1, right_wrist_y_1, left_wrist_x_1, left_wrist_y_1
    global right_wrist_x_2, right_wrist_y_2, left_wrist_x_2, left_wrist_y_2

    ret, frame = cap.read()
    if not ret:
        return left_line_1, right_line_1, left_line_2, right_line_2

    results = model(frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    frame_data = []

    for person_id, result in enumerate(results):
        keypoints = result.keypoints.xy.cpu().numpy()

        for person in keypoints:
            left_eye = person[1]
            right_eye = person[2]
            left_shoulder = person[5]
            right_shoulder = person[6]
            left_wrist = person[9]
            right_wrist = person[10]
            left_hip = person[11]
            right_hip = person[12]

            distance_between_eyes = calculate_distance(left_eye, right_eye)
            distance_between_shoulders = calculate_distance(left_shoulder, right_shoulder)
            distance_between_hips = calculate_distance(left_hip, right_hip)

            emotion = "No detectado"
            for (x, y, w, h) in faces:
                emotion = detect_emotion(frame, (x, y, w, h))
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            person_data = {
                "distancia_ojos": float(distance_between_eyes),
                "distancia_hombros": float(distance_between_shoulders),
                "distancia_cadera": float(distance_between_hips),
                "emocion": emotion
            }

            frame_data.append(person_data)

            if person_id == 0:
                left_wrist_x_1.append(left_wrist[0])
                left_wrist_y_1.append(left_wrist[1])
                right_wrist_x_1.append(right_wrist[0])
                right_wrist_y_1.append(right_wrist[1])
            elif person_id == 1:
                left_wrist_x_2.append(left_wrist[0])
                left_wrist_y_2.append(left_wrist[1])
                right_wrist_x_2.append(right_wrist[0])
                right_wrist_y_2.append(right_wrist[1])

    if frame_data:
        data_list.append(frame_data)

    annotated_frame = results[0].plot()
    cv2.imshow('Estimacion de pose', annotated_frame)

    left_line_1.set_data(left_wrist_x_1, left_wrist_y_1)
    right_line_1.set_data(right_wrist_x_1, right_wrist_y_1)
    left_line_2.set_data(left_wrist_x_2, left_wrist_y_2)
    right_line_2.set_data(right_wrist_x_2, right_wrist_y_2)

    return left_line_1, right_line_1, left_line_2, right_line_2

animation = animation.FuncAnimation(fig, graphic, interval=50, blit=True)
plt.show()

with open(json_file, "w") as file:
    json.dump(data_list, file, indent=4)

cap.release()
cv2.destroyAllWindows()