from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from numpy import cos, sin, pi
from random import random
import numpy as np

def __main__():
    show_swiss_roll()
    # show_cylinder()
    # show_digits()
    return

def show_swiss_roll():
    X, t = make_swiss_roll(n_samples=500, noise=0.0, random_state=0)
    show_3d_plot(X, t, 'Swiss Roll')
    run_Isomap(X, t, 'Swiss Roll')
    run_laplacian_eigenmaps(X, t, 'Swiss Roll')
    run_TSNE(X, t, 'Swiss Roll')
    run_autoencoder(X, t, 'Swiss Roll')

def show_cylinder():
    cylinder = [point_cylinder() for _ in range(500)]
    X = [c[:3] for c in cylinder]
    X = np.array(X)
    t = [c[3] for c in cylinder]
    show_3d_plot(X, t, 'Cylinder')
    run_Isomap(X, t, 'Cylinder')
    run_laplacian_eigenmaps(X, t, 'Cylinder')
    run_TSNE(X, t, 'Cylinder')
    run_autoencoder(X, t, 'Cylinder')

def show_digits():
    digits = load_digits()
    X = digits.data
    t = digits.target
    show_3d_plot(X, t, 'Digits')
    run_Isomap(X, t, 'Digits')
    run_laplacian_eigenmaps(X, t, 'Digits')
    run_TSNE(X, t, 'Digits')
    run_autoencoder(X, t, 'Digits')

def point_cylinder():
    theta = random() * 2 * pi
    return cos(theta), sin(theta), random(), theta * 3

def show_3d_plot(X, t, label):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.Spectral, s=20)
    ax.set_title(label)
    plt.colorbar(scatter)
    plt.show()

def show_2d_plot(X, t, label):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.Spectral, s=20)
    plt.title(label)
    plt.axis('equal')
    plt.show()

def run_TSNE(X, t, label):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    show_2d_plot(X_tsne, t, f'{label} after t-SNE')

def run_Isomap(X, t, label):
    iso = Isomap(n_components=2, n_neighbors=6)
    X_iso = iso.fit_transform(X)
    show_2d_plot(X_iso, t, f'{label} after Isomap')

def run_laplacian_eigenmaps(X, t, label):
    le = SpectralEmbedding(n_components=2, n_neighbors=6)
    X_le = le.fit_transform(X)
    show_2d_plot(X_le, t, f'{label} after Laplacian Eigenmaps')

def run_autoencoder(X, t, label):
    input_layer = Input(shape=X.shape[1])
    encoder_layer = Dense(2, activation='linear')(input_layer)
    decoder_layer = Dense(X.shape[1], activation='relu')(encoder_layer)
    autoencoder = Model(input_layer, decoder_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    X_m = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-6)
    autoencoder.fit(X_m, X_m, epochs=100, batch_size=30, shuffle=True, verbose=0)
    encoder = Model(input_layer, encoder_layer)
    X_encoded = encoder.predict(X)
    show_2d_plot(X_encoded, t, f'{label} after Autoencoder')
    

if __name__ == "__main__":
    __main__()