import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN ---
VAL_FILE = 'val_processed.csv'
MODEL_PATH = '../models/best_model.pkl'
SCALER_PATH = '../models/scaler.pkl'
COLUMNS_PATH = '../models/model_columns.pkl'
TARGET = "SalePrice"
# CAT_COLS ya no se necesita aquí porque la data ya viene transformada

def eval_model(filename):
    path = os.path.join('../data/processed', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
    df = pd.read_csv(path).set_index('Id')
    print(f'{filename} cargado correctamente | Shape: {df.shape}')
    
    # 1. Cargar Artefactos
    print('Cargando modelo y artefactos...')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH, 'rb') as f:
        model_columns = pickle.load(f)
        
    # 2. Separar X e y
    X_test = df.drop(TARGET, axis=1)
    y_test = df[TARGET]
    
    # 3. Preprocesamiento
    
    # Alineación de columnas (CRÍTICO)
    # Aun así mantenemos esto por seguridad, por si el orden cambió o falta alguna columna
    X_test = X_test.reindex(columns=model_columns, fill_value=0)
    
    # Scaling (CRÍTICO)
    # La data de validación NO estaba escalada en train.py, así que esto SÍ se ejecuta.
    # Usamos transform, NO fit.
    numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filtramos columnas que realmente existan para evitar errores
    numeric_features = [c for c in numeric_features if c in X_test.columns]
    
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # 4. Predicción
    print("Generando predicciones...")
    y_pred = model.predict(X_test)
    
    # 5. Métricas
    print("\n" + "="*30)
    print(" REPORTE DE MÉTRICAS (REGRESIÓN)")
    print("="*30)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Media del dataset de validación: {:.2f}".format(y_test.mean()))
    print(f"MAE (Error Absoluto Medio): {mae:.4f}")
    print(f"MSE (Error Cuadrático Medio): {mse:.4f}")
    print(f"RMSE (Raíz del MSE):         {rmse:.4f}")
    print(f"R2 Score (Coef. Determ.):    {r2:.4f}")
    print("-" * 30)

def main():
    eval_model(VAL_FILE)
    print('Finalizó la validación del Modelo')

if __name__ == "__main__":
    main()