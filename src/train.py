import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
INPUT_FILE = 'train_processed.csv'         # Viene de make_dataset
VAL_OUTPUT_FILE = 'val_processed.csv'      # Lo crearemos para evaluate.py
MODEL_PATH = '../models/best_model.pkl'
SCALER_PATH = '../models/scaler.pkl'
COLUMNS_PATH = '../models/model_columns.pkl'

# Columnas categóricas (Mismas que en make_dataset)
CAT_COLS = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]
TARGET = "SalePrice"

def load_data(filename):
    path = os.path.join('../data/processed', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_csv(path).set_index('Id')
    print(f"{filename} cargado | Shape: {df.shape}")
    return df

def main():
    print("--- INICIANDO PROCESO DE ENTRENAMIENTO ---")

    # 1. Cargar Datos
    df = load_data(INPUT_FILE)

    # 2. Separar Features y Target
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 3. One Hot Encoding (Fit)
    # Al hacer get_dummies aquí, definimos la estructura "oficial" del modelo
    X = pd.get_dummies(X, columns=CAT_COLS)
    
    # Guardamos la lista de columnas final para replicarla en inferencia/evaluación
    final_columns = X.columns.tolist()

    # 4. Split Train/Test (80/20)
    # X_test y y_test NO se usarán para entrenar, se guardarán para evaluate.py
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Scaling (Fit solo en Train)
    # Seleccionamos columnas numéricas (excluyendo las binarias de dummies si se desea, 
    # pero para simplificar escalaremos las numéricas originales detectadas)
    important_num_cols = [col for col in X_train.columns if col not in final_columns or X_train[col].nunique() > 2]
    # O simplemente re-detectamos numéricas:
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Filtramos para asegurarnos que existen (algunas dummies son uint8)
    numeric_features = [c for c in numeric_features if c in X_train.columns]

    scaler = StandardScaler()
    # Ajustamos el scaler SOLO con el set de entrenamiento
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    # 6. Entrenamiento
    print("Entrenando XGBRegressor...")
    xgb_mod = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42, n_jobs=-1)
    xgb_mod.fit(X_train, y_train)
    print('Modelo entrenado exitosamente')

    # 7. Guardar Artefactos
    # a) Modelo
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_mod, f)
    
    # b) Scaler (Para usarlo en evaluate y predict)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    # c) Columnas (Para alineación)
    with open(COLUMNS_PATH, 'wb') as f:
        pickle.dump(final_columns, f)

    # d) Data de Validación (Para que evaluate.py tenga qué evaluar)
    # Reconstruimos el dataframe de validación tal cual salió del split (sin escalar aun)
    val_df = X_val.copy()
    val_df[TARGET] = y_val
    val_df.to_csv(os.path.join('../data/processed', VAL_OUTPUT_FILE))
    
    print(f"Artefactos guardados en ../models/")
    print(f"Set de validación guardado en ../data/processed/{VAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()