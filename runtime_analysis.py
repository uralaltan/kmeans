import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
import time
import os
from scipy.cluster.hierarchy import linkage


# Runtime klasörünü oluştur
def create_runtime_directory(directory_name="runtime"):
    # Eğer klasör yoksa oluştur
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"'{directory_name}' klasörü oluşturuldu.")
    else:
        print(f"'{directory_name}' klasörü zaten mevcut.")

    return directory_name


# Rastgele veri oluştur
def generate_sample_data(n_samples=500, n_features=2, centers=3, random_state=42):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=centers, random_state=random_state)
    return X, y


# K-means kümeleme runtime analizi
def kmeans_runtime_analysis(sample_sizes, n_clusters=3, n_runs=5):
    kmeans_times = []

    for size in sample_sizes:
        size_times = []

        for _ in range(n_runs):  # Her boyut için n_runs kez çalıştır (ortalama için)
            X, _ = generate_sample_data(n_samples=size, centers=n_clusters)

            # K-means'i çalıştır ve süreyi ölç
            start_time = time.time()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)
            end_time = time.time()

            size_times.append(end_time - start_time)

        # Bu boyut için ortalama süreyi hesapla
        kmeans_times.append(np.mean(size_times))

    return kmeans_times


# Karmaşıklık yorumlaması için yardımcı fonksiyon
def get_complexity_notation(slope):
    if slope < 1.25:
        return "O(n)"
    elif slope < 1.75:
        return "O(n log n)"
    elif slope < 2.25:
        return "O(n²)"
    elif slope < 2.75:
        return "O(n² log n)"
    else:
        return "O(n³)"


# Hiyerarşik kümeleme runtime analizi
def hierarchical_runtime_analysis(sample_sizes, n_clusters=3, n_runs=5):
    hierarchical_times = []
    linkage_times = []

    for size in sample_sizes:
        size_times_h = []
        size_times_l = []

        for _ in range(n_runs):  # Her boyut için n_runs kez çalıştır
            X, _ = generate_sample_data(n_samples=size, centers=n_clusters)

            # Scipy linkage ile hiyerarşik kümeleme süresini ölç
            start_time = time.time()
            Z = linkage(X, method='ward')
            end_time = time.time()
            size_times_l.append(end_time - start_time)

            # AgglomerativeClustering ile hiyerarşik kümeleme süresini ölç
            start_time = time.time()
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            hierarchical.fit(X)
            end_time = time.time()

            size_times_h.append(end_time - start_time)

        # Bu boyut için ortalama süreleri hesapla
        hierarchical_times.append(np.mean(size_times_h))
        linkage_times.append(np.mean(size_times_l))

    return hierarchical_times, linkage_times


# Runtime analizini görselleştir ve kaydet
def plot_runtime_comparison(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                            save_path="runtime/runtime_comparison.png"):
    plt.figure(figsize=(12, 8))

    plt.plot(sample_sizes, kmeans_times, 'o-', color='blue', label='K-means')
    plt.plot(sample_sizes, hierarchical_times, 's-', color='red', label='AgglomerativeClustering')
    plt.plot(sample_sizes, linkage_times, '^-', color='green', label='Scipy Linkage')

    plt.title('K-means vs. Hiyerarşik Kümeleme Runtime Analizi', fontsize=16)
    plt.xlabel('Örnek Sayısı', fontsize=14)
    plt.ylabel('Çalışma Süresi (saniye)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Runtime karşılaştırma grafiği kaydedildi: {save_path}")


# Her algoritmanın ayrı ayrı görselleştirilmesi
def plot_individual_runtimes(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                             folder_path="runtime"):
    # K-means
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, kmeans_times, 'o-', color='blue', linewidth=2)
    plt.title('K-means Runtime Analizi', fontsize=16)
    plt.xlabel('Örnek Sayısı', fontsize=14)
    plt.ylabel('Çalışma Süresi (saniye)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    kmeans_path = f"{folder_path}/kmeans_runtime.png"
    plt.savefig(kmeans_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K-means runtime grafiği kaydedildi: {kmeans_path}")

    # Hiyerarşik kümeleme
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, hierarchical_times, 's-', color='red', linewidth=2,
             label='AgglomerativeClustering')
    plt.plot(sample_sizes, linkage_times, '^-', color='green', linewidth=2,
             label='Scipy Linkage')
    plt.title('Hiyerarşik Kümeleme Runtime Analizi', fontsize=16)
    plt.xlabel('Örnek Sayısı', fontsize=14)
    plt.ylabel('Çalışma Süresi (saniye)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    hierarchical_path = f"{folder_path}/hierarchical_runtime.png"
    plt.savefig(hierarchical_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hiyerarşik kümeleme runtime grafiği kaydedildi: {hierarchical_path}")


# Logaritmik ölçekte çizim yap
def plot_log_scale_runtime(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                           save_path="runtime/runtime_log_scale.png"):
    plt.figure(figsize=(12, 8))

    plt.loglog(sample_sizes, kmeans_times, 'o-', color='blue', label='K-means')
    plt.loglog(sample_sizes, hierarchical_times, 's-', color='red', label='AgglomerativeClustering')
    plt.loglog(sample_sizes, linkage_times, '^-', color='green', label='Scipy Linkage')

    plt.title('K-means vs. Hiyerarşik Kümeleme Runtime Analizi (Log Ölçek)', fontsize=16)
    plt.xlabel('Örnek Sayısı (log ölçek)', fontsize=14)
    plt.ylabel('Çalışma Süresi (saniye, log ölçek)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Teorik karmaşıklık çizgileri
    x = np.array(sample_sizes)

    # K-means için O(n) çizgisi (n = örnek sayısı)
    scale_factor_kmeans = kmeans_times[-1] / x[-1]
    y_kmeans = scale_factor_kmeans * x
    plt.loglog(x, y_kmeans, '--', color='blue', alpha=0.5, label='O(n) - Teorik')

    # Hiyerarşik için O(n²) çizgisi
    scale_factor_hier = hierarchical_times[-1] / (x[-1] ** 2)
    y_hier = scale_factor_hier * (x ** 2)
    plt.loglog(x, y_hier, '--', color='red', alpha=0.5, label='O(n²) - Teorik')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Log ölçekli runtime karşılaştırma grafiği kaydedildi: {save_path}")


# Karmaşıklık özeti grafiği
def plot_complexity_summary(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                            save_path="runtime/complexity_summary.png"):
    # Karmaşıklık tahminleri için regresyon yapma
    from sklearn.linear_model import LinearRegression

    # Log-log için veriler
    log_sizes = np.log(sample_sizes)
    log_kmeans = np.log(kmeans_times)
    log_hierarchical = np.log(hierarchical_times)
    log_linkage = np.log(linkage_times)

    # Eğimler hesaplanıyor (karmaşıklığı temsil eder)
    reg_kmeans = LinearRegression().fit(log_sizes.reshape(-1, 1), log_kmeans)
    reg_hierarchical = LinearRegression().fit(log_sizes.reshape(-1, 1), log_hierarchical)
    reg_linkage = LinearRegression().fit(log_sizes.reshape(-1, 1), log_linkage)

    kmeans_slope = reg_kmeans.coef_[0]
    hierarchical_slope = reg_hierarchical.coef_[0]
    linkage_slope = reg_linkage.coef_[0]

    # Sonuçlar
    plt.figure(figsize=(10, 8))
    algorithms = ['K-means', 'AgglomerativeClustering', 'Scipy Linkage']
    slopes = [kmeans_slope, hierarchical_slope, linkage_slope]
    colors = ['blue', 'red', 'green']

    bars = plt.bar(algorithms, slopes, color=colors, alpha=0.7)

    plt.title('Kümeleme Algoritmalarının Tahmini Zaman Karmaşıklığı', fontsize=16)
    plt.ylabel('Eğim (Log-Log Regresyon)', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Değerleri göster
    for bar, slope in zip(bars, slopes):
        plt.text(bar.get_x() + bar.get_width() / 2.,
                 slope + 0.05,
                 f'O(n^{slope:.2f})',
                 ha='center', va='bottom',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Karmaşıklık özeti grafiği kaydedildi: {save_path}")

    return kmeans_slope, hierarchical_slope, linkage_slope


# Ana fonksiyon
if __name__ == "__main__":
    # Runtime klasörünü oluştur
    folder_path = create_runtime_directory()

    # Analiz için örnek boyutları (küçük boyutlardan başla büyük boyutlara kadar)
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

    # Runtime analizlerini yap
    print("K-means runtime analizi yapılıyor...")
    kmeans_times = kmeans_runtime_analysis(sample_sizes)

    print("Hiyerarşik kümeleme runtime analizi yapılıyor...")
    hierarchical_times, linkage_times = hierarchical_runtime_analysis(sample_sizes)

    # Sonuçları tablo halinde göster
    print("\nRuntime Sonuçları (saniye):")
    print("-" * 80)
    print(f"{'Örnek Sayısı':<15} {'K-means':<15} {'Agglomerative':<15} {'Linkage':<15}")
    print("-" * 80)

    for i, size in enumerate(sample_sizes):
        print(f"{size:<15} {kmeans_times[i]:<15.5f} {hierarchical_times[i]:<15.5f} {linkage_times[i]:<15.5f}")

    print("-" * 80)

    # Grafikleri çiz ve kaydet
    print("\nGrafikler oluşturuluyor...")

    # Normal ölçekli karşılaştırma grafiği
    plot_runtime_comparison(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                            save_path=f"{folder_path}/runtime_comparison.png")

    # Her algoritma için ayrı grafikler
    plot_individual_runtimes(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                             folder_path=folder_path)

    # Log ölçeği grafiği
    plot_log_scale_runtime(sample_sizes, kmeans_times, hierarchical_times, linkage_times,
                           save_path=f"{folder_path}/runtime_log_scale.png")

    # Karmaşıklık özeti
    kmeans_slope, hierarchical_slope, linkage_slope = plot_complexity_summary(
        sample_sizes, kmeans_times, hierarchical_times, linkage_times,
        save_path=f"{folder_path}/complexity_summary.png")

    # Büyük-O notasyonu yorumu
    print("\nZaman Karmaşıklığı Analizi:")
    print(f"K-means tahmini karmaşıklık: O(n^{kmeans_slope:.2f}) ~ {get_complexity_notation(kmeans_slope)}")
    print(
        f"AgglomerativeClustering tahmini karmaşıklık: O(n^{hierarchical_slope:.2f}) ~ {get_complexity_notation(hierarchical_slope)}")
    print(f"Scipy Linkage tahmini karmaşıklık: O(n^{linkage_slope:.2f}) ~ {get_complexity_notation(linkage_slope)}")
