import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import torch
from torch.utils.data import DataLoader, Dataset

# 数据加载与预处理
def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"文件路径 '{data_path}' 不存在，请检查路径是否正确。")
    else:
        print(f"文件路径 '{data_path}' 已找到，正在加载数据...")
        df = pd.read_csv(data_path, header=None)
        return df

# 数据预处理
def preprocess_data(df):
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    std_dev = df.std()
    constant_columns = std_dev[std_dev == 0].index
    df.drop(columns=constant_columns, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

# 使用 DBSCAN 进行聚类
def cluster_data(data, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_data)
    clusters = dbscan.labels_
    return clusters

# 生成动态图数据和标签
def generate_dynamic_graph(data, lookback, lookforward, eps=0.5, min_samples=5):
    n = len(data)
    graph_data = []
    graph_labels = []
    for i in range(lookback, n - lookforward):
        window_data = data.iloc[i - lookback:i + lookforward]
        clusters = cluster_data(window_data, eps=eps, min_samples=min_samples)
        cluster_indices = np.where(clusters == clusters[-1])[0]

        if len(cluster_indices) < 2:
            continue

        current_data = window_data.iloc[cluster_indices].iloc[-lookback:].values
        future_data = window_data.iloc[-lookforward:].values.mean(axis=0)

        current_data[:, :-3][np.isnan(current_data[:, :-3])] = 0
        if np.std(current_data[:, :-3]) == 0:
            continue

        local_corr_matrix = np.corrcoef(current_data[:, :-3], rowvar=False)
        local_corr_matrix = np.nan_to_num(local_corr_matrix)
        np.fill_diagonal(local_corr_matrix, 0)

        window_data.values[:, :-3][np.isnan(window_data.values[:, :-3])] = 0
        if np.std(window_data.values[:, :-3]) == 0:
            continue

        global_corr_matrix = np.corrcoef(window_data.values[:, :-3], rowvar=False)
        global_corr_matrix = np.nan_to_num(global_corr_matrix)
        np.fill_diagonal(global_corr_matrix, 0)

        combined_corr_matrix = 0.5 * local_corr_matrix + 0.5 * global_corr_matrix
        flat_corr = combined_corr_matrix.flatten()
        extra_features = current_data[-1, -3:].reshape(-1)
        combined_features = np.concatenate([flat_corr, extra_features])

        graph_data.append(combined_features)
        graph_labels.append(future_data[:12])

    return np.array(graph_data, dtype=np.float32), np.array(graph_labels, dtype=np.float32)

# 自定义数据集
class TrafficDataset(Dataset):
    def __init__(self, graph_data, labels):
        self.graph_data = torch.tensor(graph_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx], self.labels[idx]

# 创建数据加载器
def create_dataloaders(graph_data, graph_labels, batch_size=10):
    dataset = TrafficDataset(graph_data, graph_labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
