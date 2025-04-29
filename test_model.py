import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# === Cargar modelo y codificador ===
model = joblib.load("xgboost_model_filtrado.pkl")
label_encoder = joblib.load("label_encoder_filtrado.pkl")

# === Inicializar MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Configuración de detección
gesture_queue = deque(maxlen=10)
required_frames = 7
confidence_threshold = 0.75

# Cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture = None
    confidence = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer 21 puntos (x, y, z) = 63 valores
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                X_input = np.array(landmarks).reshape(1, -1)
                probs = model.predict_proba(X_input)[0]
                pred_index = np.argmax(probs)
                confidence = probs[pred_index]

                if confidence >= confidence_threshold:
                    gesture = label_encoder.inverse_transform([pred_index])[0]
                    gesture_queue.append(gesture)

                # Mostrar predicción
                cv2.putText(frame, f"{label_encoder.inverse_transform([pred_index])[0]} ({confidence:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar gesto confirmado si aparece varias veces
    if gesture and gesture_queue.count(gesture) >= required_frames:
        cv2.putText(frame, f"Confirmado: {gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Detección de Señas (5 gestos)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
