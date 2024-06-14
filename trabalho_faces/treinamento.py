import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle  # Importar o módulo pickle

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Função para extrair landmarks faciais
def get_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_points = []
        for lm in landmarks.landmark:
            landmark_points.append([lm.x, lm.y, lm.z])
        return np.array(landmark_points).flatten()
    return None

# Carregar imagens do dataset e extrair landmarks
dataset_dir = 'images'
landmarks_data = []
labels = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image = cv2.imread(os.path.join(dataset_dir, filename))
        landmarks = get_landmarks(image)
        if landmarks is not None:
            landmarks_data.append(landmarks)
            labels.append(filename.split('_')[0])  # Supondo que o nome do arquivo começa com o rótulo
        else:
            print(f"Falha ao extrair landmarks para {filename}")

landmarks_data = np.array(landmarks_data)
labels = np.array(labels)

# Verificar distribuição dos rótulos
print("Distribuição dos rótulos:")
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

# Codificar os rótulos
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(landmarks_data, labels, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ajustar os hiperparâmetros do SVM
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2 ,cv=5)
grid.fit(X_train, y_train)

# Avaliar o modelo ajustado
y_pred = grid.predict(X_test)
print(f'Acurácia: {accuracy_score(y_test, y_pred)}')

with open('modelo_svm.pkl', 'wb') as f:
    pickle.dump(grid, f)

