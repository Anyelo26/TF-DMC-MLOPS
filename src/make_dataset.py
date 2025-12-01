import pandas as pd
import os

# Columnas categóricas predefinidas para incluir siempre
CAT_COLS = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]
TARGET  = ["SalePrice"]

def read_file_csv(filename):
    path = os.path.join('../data/raw/', filename)
    # Agregamos validación básica
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path).set_index('Id')
    print(f'{filename} cargado correctamente | Shape: {df.shape}')
    return df

def data_cleaning(df):
    df_cleaned = df.copy()
    
    # 1. FILTRO DE FILAS (Variable Objetivo: SalePrice)
    if TARGET[0] in df_cleaned.columns:
        # Primero eliminamos los Nulos (NaN) en esa columna específica
        df_cleaned = df_cleaned.dropna(subset=[TARGET[0]])
        
        # Luego eliminamos los menores o iguales a 0 (negativos y ceros)
        df_cleaned = df_cleaned[df_cleaned[TARGET[0]] > 0]
        
        print(f"Filas filtradas por SalePrice válido")
    
    return df_cleaned

def get_best_features(df_train, target_col=TARGET[0], threshold=0.5):
    """
    Esta función SOLO se usa con el set de entrenamiento para DECIDIR qué columnas usar.
    """
    # 1. Correlación numérica
    # Seleccionamos solo numéricas para evitar errores en .corr()
    numeric_df = df_train.select_dtypes(include=['number'])
    corr = numeric_df.corr()[target_col]
    
    # Filtramos por umbral positivo o negativo
    important_num_cols = list(corr[(corr > threshold) | (corr < -threshold)].index)
    
    # Aseguramos que el target esté en la lista para no perderlo en el train
    if target_col in important_num_cols:
        important_num_cols.remove(target_col)
        
    # 2. Unimos con las categóricas predefinidas
    selected_features = important_num_cols + CAT_COLS
    print(f"Features seleccionados ({len(selected_features)}): {selected_features}")
    return selected_features

def data_export(df, filename):
    output_path = os.path.join('../data/processed/', filename)
    df.to_csv(output_path)
    print(f'{filename} exportado correctamente | Shape: {df.shape}')

def main():
    
    # 1. Dataset para Proceso de Entrenamiento (Master)
    print("--- Procesando TRAIN ---")
    df_tr = read_file_csv("train.csv")
    df_tr_cleaned = data_cleaning(df_tr)
    
    # seleccionamos los mejores features
    best_features = get_best_features(df_tr_cleaned)
    # Transformamos Train usando esas columnas
    data_export(df_tr_cleaned[best_features + TARGET], "train_processed.csv")
    
    
    # 2. Dataset para Proceso de Test (Dependiente del Train)
    print("\n--- Procesando TEST ---")
    df_te = read_file_csv("test.csv")
    df_te_cleaned = data_cleaning(df_te)

    # Transformamos Test usando LA MISMA LISTA 'best_features' obtenida arriba
    # aqui no se incluye la variable target porque se requiere inferencia con data nueva
    data_export(df_te_cleaned[best_features], "test_processed.csv")

if __name__ == "__main__":
    main()