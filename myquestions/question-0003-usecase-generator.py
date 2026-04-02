import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

def normalizar_min_max():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función normalizar_min_max.
    """

    n = random.randint(5, 15)
    n_cols = random.randint(2, 4)

    data = np.random.randn(n, n_cols) * 50 + 100
    columnas = [f'col_{i}' for i in range(n_cols)]

    df = pd.DataFrame(data, columns=columnas)

    input_data = {
        'df': df.copy(),
        'columnas': columnas
    }

    df_out = df.copy()

    scaler = MinMaxScaler()
    df_out[columnas] = scaler.fit_transform(df_out[columnas])

    return input_data, df_out


if __name__ == "__main__":
    i, o = normalizar_min_max()

    print("---- inputs ----")
    for k, v in i.items():
        print("\n", k, ":\n", v)

    print("\n---- expected output ----\n", o)
