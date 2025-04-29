import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Cargar el CSV completo
df = pd.read_csv("hand_landmarks_data.csv")

# Filtrar solo las clases necesarias
clases_deseadas = ["mute", "dislike", "peace", "palm", "call"]
df = df[df["label"].isin(clases_deseadas)]

# Separar características y etiquetas
X = df.drop("label", axis=1).values
y = df["label"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Entrenar modelo con número reducido de árboles para velocidad
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_estimators=50)
model.fit(X_train, y_train)

# Guardar modelo y codificador
joblib.dump(model, "xgboost_model_filtrado.pkl")
joblib.dump(label_encoder, "label_encoder_filtrado.pkl")

# Evaluación opcional
accuracy = model.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
