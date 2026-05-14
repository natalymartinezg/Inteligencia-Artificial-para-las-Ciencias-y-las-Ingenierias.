import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def proyectar_y_reconstruir_pca(X, n_componentes):

    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X)

    pca = PCA(
        n_components=n_componentes,
        svd_solver='full'
    )

    X_proyectada = pca.fit_transform(X_escalada)

    X_reconstruida_escalada = pca.inverse_transform(X_proyectada)

    X_reconstruida = scaler.inverse_transform(
        X_reconstruida_escalada
    )

    varianza_explicada = pca.explained_variance_ratio_

    return (
        X_proyectada,
        X_reconstruida,
        varianza_explicada
    )
