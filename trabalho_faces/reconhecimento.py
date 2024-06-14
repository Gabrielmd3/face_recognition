import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle  # Importar o módulo pickle

# Função para reconhecer o rosto em tempo real
def recognize_face(frame):
    landmarks = get_landmarks(frame)
    if landmarks is not None:
        landmarks = landmarks.reshape(1, -1)
        prob = grid.predict_proba(landmarks)
        label = grid.predict(landmarks)
        if prob[0][label] > 0.5:  # Ajuste o limiar conforme necessário
            return label_encoder.inverse_transform(label)[0]
    return None

# Capturar vídeo da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    name = recognize_face(frame)
    if name:
        cv2.putText(frame, f'Recognized: {name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Not Recognized', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
