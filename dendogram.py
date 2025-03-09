import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import os


# Dendogram klasörünü oluştur
def create_dendogram_directory(directory_name="dendogram"):
    # Eğer klasör yoksa oluştur
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"'{directory_name}' klasörü oluşturuldu.")
    else:
        print(f"'{directory_name}' klasörü zaten mevcut.")

    return directory_name


# Rastgele veri oluştur
def generate_sample_data(n_samples=50, n_features=2, centers=3, random_state=42):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=centers, random_state=random_state)
    return X, y


# Dendogram çiz ve kaydet
def plot_dendrogram(X, method='ward', figsize=(10, 6),
                    title="Hiyerarşik Kümeleme Dendogramı",
                    save_path="/dendogram/dendogram_basic.png"):
    # Bağlantı matrisini hesapla
    Z = linkage(X, method=method)

    # Görselleştirme
    plt.figure(figsize=figsize)
    plt.title(title)

    # Dendogram çizimi
    dendrogram(Z, leaf_rotation=90., leaf_font_size=10.)

    plt.xlabel('Örnek İndeksi')
    plt.ylabel('Mesafe')
    plt.tight_layout()

    # Resmi kaydet
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Dendogram kaydedildi: {save_path}")

    return Z


# Farklı bağlantı yöntemleri ile dendogramları göster ve kaydet
def show_different_linkage_methods(X, save_path="/dendogram/dendogram_methods.png"):
    methods = ['single', 'complete', 'average', 'ward']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        Z = linkage(X, method=method)
        ax = axes[i]
        dendrogram(Z, ax=ax, leaf_rotation=90., leaf_font_size=8.)
        ax.set_title(f'Bağlantı Metodu: {method}')
        ax.set_xlabel('Örnek İndeksi')
        ax.set_ylabel('Mesafe')

    plt.tight_layout()

    # Resmi kaydet
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Karşılaştırmalı dendogram kaydedildi: {save_path}")


# Ana işlem
if __name__ == "__main__":
    # Dendogram klasörünü oluştur
    folder_path = create_dendogram_directory()

    # Veri oluştur
    X, y = generate_sample_data(n_samples=40, centers=4)

    # Temel dendogram çizimi ve kaydetme
    Z = plot_dendrogram(X, method='ward',
                        save_path=f"{folder_path}/dendogram_basic.png")

    # Farklı bağlantı yöntemleri ile dendogramlar
    show_different_linkage_methods(X,
                                   save_path=f"{folder_path}/dendogram_methods.png")

    # Gerçek veri kullanma örneği
    from sklearn import datasets

    # İris veri setini yükle
    iris = datasets.load_iris()
    X_iris = iris.data

    plt.figure(figsize=(12, 8))
    plt.title("İris Veri Seti Hiyerarşik Kümeleme Dendogramı")
    dendrogram(linkage(X_iris, method='ward'), leaf_rotation=90., leaf_font_size=8.)
    plt.xlabel('Örnek İndeksi')
    plt.ylabel('Mesafe')
    plt.tight_layout()

    # İris dendogramını kaydet
    iris_path = f"{folder_path}/dendogram_iris.png"
    plt.savefig(iris_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"İris dendogramı kaydedildi: {iris_path}")

    # Farklı veri boyutlarıyla dendogramlar
    for n_samples in [20, 50, 100]:
        X, _ = generate_sample_data(n_samples=n_samples, centers=3)

        plt.figure(figsize=(10, 6))
        plt.title(f"Dendogram (n={n_samples})")
        dendrogram(linkage(X, method='ward'), leaf_rotation=90., leaf_font_size=8.)
        plt.xlabel('Örnek İndeksi')
        plt.ylabel('Mesafe')
        plt.tight_layout()

        size_path = f"{folder_path}/dendogram_n{n_samples}.png"
        plt.savefig(size_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dendogram (n={n_samples}) kaydedildi: {size_path}")