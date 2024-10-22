import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import load_data, preprocess_data, generate_dynamic_graph, create_dataloaders
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt

# Neural Network 定义
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(64, 12)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.output(x)

# 主函数
def main():
    data_path = 'DATA/PEMS_BAY.csv'

    # 数据加载与预处理
    df = load_data(data_path)
    df = preprocess_data(df)

    # 生成动态图数据
    lookback_days = 12
    lookforward_days = 12
    graph_data, graph_labels = generate_dynamic_graph(df, lookback_days, lookforward_days)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(graph_data, graph_labels, batch_size=10)

    # 模型定义
    input_dim = graph_data.shape[1]
    model = NeuralNetwork(input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_fn, device)

    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pth'))
    mse, rmse, mae, mape = evaluate_model(model, test_loader, loss_fn, device)
    print(f'Test MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape:.2f}%')

    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
