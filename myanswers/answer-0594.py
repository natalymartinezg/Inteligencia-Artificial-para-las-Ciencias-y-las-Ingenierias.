import pandas as pd
import numpy as np

def extraer_features_temporales(df, col_fecha):
    
    # 1. Copia del DataFrame
    df_res = df.copy()
    
    # 2. Convertir a datetime
    fechas = pd.to_datetime(df_res[col_fecha])
    
    # 3. Extraer features
    df_res['mes'] = fechas.dt.month
    df_res['dia_semana'] = fechas.dt.dayofweek
    
    # 4. Fin de semana
    df_res['es_fin_de_semana'] = df_res['dia_semana'].isin([5, 6]).astype(int)
    
    # 5. Eliminar columna original
    df_res = df_res.drop(columns=[col_fecha])
    
    return df_res
