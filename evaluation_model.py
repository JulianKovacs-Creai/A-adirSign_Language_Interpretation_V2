import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
from collections import deque

# === Cargar modelo y codificador ===
model = joblib.load("xgboost_model_filtrado.pkl")
label_encoder = joblib.load("label_encoder_filtrado.pkl")

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Mapeo de acciones ===
gesture_actions = {
    "mute": ("Silenciar micrófono", lambda: pyautogui.hotkey("ctrl", "shift", "m")),
    "dislike": ("Apagar cámara", lambda: pyautogui.hotkey("ctrl", "shift", "o")),
    "peace": ("Activar micrófono y cámara", lambda: (
        pyautogui.hotkey("ctrl", "shift", "m"),
        pyautogui.hotkey("ctrl", "shift", "o")
    )),
    "palm": ("Levantar la mano", lambda: pyautogui.hotkey("ctrl", "shift", "k")),
    "call": ("Cortar llamada", lambda: pyautogui.hotkey("ctrl", "shift", "h")),  
}

# Configuración de control
gesture_queue = deque(maxlen=10)
required_frames = 10
confidence_threshold = 0.78
last_display_text = ""
last_gesture = ""

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

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                X_input = np.array(landmarks).reshape(1, -1)
                probs = model.predict_proba(X_input)[0]
                pred_index = np.argmax(probs)
                confidence = probs[pred_index]
                gesture = label_encoder.inverse_transform([pred_index])[0]

                if confidence >= confidence_threshold:
                    gesture_queue.append(gesture)

                # Mostrar predicción actual
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Ejecutar acción si se repite varias veces
    if gesture and gesture_queue.count(gesture) >= required_frames:
        if gesture in gesture_actions:
            text, action_func = gesture_actions[gesture]
            action_func()
            last_display_text = text
            last_gesture = gesture
            gesture_queue.clear()

    # Mostrar última acción ejecutada
    if last_display_text:
        cv2.putText(frame, last_display_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    cv2.imshow("Control Teams por Señas (5 gestos)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
