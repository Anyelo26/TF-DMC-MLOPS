import pandas as pd
import xgboost as xgb
import pickle
import os

# --- CONFIGURACIÓN ---
INPUT_FILE = 'test_processed.csv'
OUTPUT_FILE = 'submission_final.csv'
OUTPUT_DIR = '../data/scores/'

MODEL_PATH = '../models/best_model.pkl'
SCALER_PATH = '../models/scaler.pkl'
COLUMNS_PATH = '../models/model_columns.pkl'

CAT_COLS = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]

def score_model(filename, scores_file):
    # 1. Cargar Data
    input_path = os.path.join('../data/processed', filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el archivo: {input_path}")
    
    df = pd.read_csv(input_path).set_index('Id')
    print(f'{filename} cargado correctamente | Shape original: {df.shape}')

    # 2. Cargar Artefactos
    print('Cargando modelo, scaler y columnas...')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH, 'rb') as f:
        model_columns = pickle.load(f)

    # ---------------------------------------------------------
    # 3. PREPROCESAMIENTO
    # ---------------------------------------------------------
    
    # A) One Hot Encoding
    df_processed = pd.get_dummies(df, columns=CAT_COLS)
    
    # B) Alineación de Columnas (Reindex)
    # Esto asegura que tengas TODAS las columnas que espera el modelo (incluyendo las dummies)
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)
    
    # C) Scaling 
    # En lugar de seleccionar por tipo, LE PREGUNTAMOS al scaler qué columnas conoce.
    # Esto evita que se escale columnas dummies (Heating_Floor, etc.) que no deben escalarse.
    
    if hasattr(scaler, 'feature_names_in_'):
        features_to_scale = scaler.feature_names_in_
    else:
        print("Advertencia: Scaler antiguo, intentando inferir numéricas...")
        features_to_scale = df_processed.select_dtypes(include=['float64', 'int64']).columns
    
    print(f"Escalando {len(features_to_scale)} variables (según el scaler)...")
    
    # Aplicamos transform SOLO a las columnas que el scaler conoce
    df_processed[features_to_scale] = scaler.transform(df_processed[features_to_scale])
        
    print(f'Data procesada lista para inferencia | Shape final: {df_processed.shape}')

    # ---------------------------------------------------------
    # 4. PREDICCIÓN
    # ---------------------------------------------------------
    preds = model.predict(df_processed)
    
    # 5. Exportar Resultados
    submission = pd.DataFrame({
        'Id': df.index,
        'SalePrice': preds
    })
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, scores_file)
    submission.to_csv(output_path, index=False)
    
    print(f"Predicciones exportadas exitosamente en: {output_path}")

def main():
    score_model(INPUT_FILE, OUTPUT_FILE)
    print('Finalizó el Scoring del Modelo')

if __name__ == "__main__":
    main()