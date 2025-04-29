# 🖐️ Hand Gesture Control for Microsoft Teams

This project allows real-time control of **Microsoft Teams** using **hand gestures** captured via webcam. The system recognizes 5 specific gestures and triggers corresponding actions like muting the microphone, turning the camera on/off, raising your hand, or finish a call — all without touching your keyboard.

---

## 🎯 Features

| Gesture   | Action                                  | Shortcut in Teams            |
|-----------|------------------------------------------|-------------------------------|
| `mute`    | Toggle microphone                        | `Ctrl + Shift + M`            |
| `dislike` | Toggle camera                            | `Ctrl + Shift + O`            |
| `peace`   | Toggle both mic and camera               | `Ctrl + Shift + M` + `O`      |
| `palm`    | Raise hand                               | `Ctrl + Shift + K`            |
| `call`    | Exit call                                | `Ctrl + Shift + H`            |

---

## 📦 Tech Stack

- **Python 3**
- [MediaPipe](https://google.github.io/mediapipe/) – hand landmark detection
- [XGBoost](https://xgboost.readthedocs.io/) – gesture classification
- **PyAutoGUI** – simulate keyboard shortcuts
- **OpenCV** – real-time webcam feed
- **scikit-learn** – label encoding and dataset split

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe xgboost scikit-learn pyautogui joblib pandas
```

### 2. Run the Script

Make sure the following files are in the same directory:
- `xgboost_model_filtrado.pkl`
- `label_encoder_filtrado.pkl`

Then run:

```bash
python evaluation_model.py
```

> Press `ESC` to exit the application.

---

## 🧪 Training Your Own Model

You can retrain the model using:

```bash
python entrenar_modelo.py
```

Make sure to provide a properly labeled CSV with MediaPipe landmarks for gestures like `mute`, `dislike`, `peace`, `palm`, and `call`.

---

## 📂 File Structure

```
├── evaluation_model.py         # Main gesture-to-Teams controller
├── train_model.py              # Training script for XGBoost
├── xgboost_model_filtrado.pkl      # Trained classifier
├── label_encoder_filtrado.pkl      # Label encoder for gestures
├── hand_landmarks_data.csv         # Original dataset
```

