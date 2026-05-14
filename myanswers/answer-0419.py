import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def proyectar_y_reconstruir_pca(X, n_componentes):
    """
    Estandariza una matriz X, aplica PCA, reconstruye los datos
    y devuelve la proyección, la reconstrucción y la varianza explicada.

    Parámetros
    ----------
    X : numpy.ndarray
        Matriz numérica bidimensional.
    
    n_componentes : int
        Número de componentes principales.

    Retorna
    -------
    tuple
        (
            X_proyectada,
            X_reconstruida,
            varianza_explicada
        )
    """

    # 1. Estandarizar datos
    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X)

    # 2. Ajustar PCA
    pca = PCA(
        n_components=n_componentes,
        svd_solver='full'
    )

    # 3. Obtener proyección reducida
    X_proyectada = pca.fit_transform(X_escalada)

    # 4. Reconstruir datos
    X_reconstruida_escalada = pca.inverse_transform(X_proyectada)

    X_reconstruida = scaler.inverse_transform(
        X_reconstruida_escalada
    )

    # 5. Obtener varianza explicada
    varianza_explicada = pca.explained_variance_ratio_

    # 6. Retornar resultados
    return (
        X_proyectada,
        X_reconstruida,
        varianza_explicada
    )


# ==================================================
# EJEMPLO DE VALIDACIÓN
# ==================================================

if __name__ == "__main__":

    # Datos de prueba
    X = np.random.rand(20, 5)

    resultado = proyectar_y_reconstruir_pca(
        X,
        n_componentes=2
    )

    X_proyectada, X_reconstruida, varianza = resultado

    print("=== RESULTADO ===")

    print("\nX_proyectada:")
    print(X_proyectada)

    print("\nShape X_proyectada:")
    print(X_proyectada.shape)

    print("\nX_reconstruida:")
    print(X_reconstruida)

    print("\nShape X_reconstruida:")
    print(X_reconstruida.shape)

    print("\nVarianza explicada:")
    print(varianza)
