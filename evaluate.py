import torch
import numpy as np

# 模型评估函数
def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    mse = np.mean((predictions - ground_truths) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truths))

    non_zero_indices = ground_truths != 0
    mape = np.mean(np.abs(
        (ground_truths[non_zero_indices] - predictions[non_zero_indices]) / ground_truths[non_zero_indices])) * 100

    return mse, rmse, mae, mape
