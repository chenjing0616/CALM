import numpy as np
import torch


# Helper function to calculate model evaluation metrics
def calculate_metrics(predictions, ground_truths):
    mse = np.mean((predictions - ground_truths) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truths))

    non_zero_indices = ground_truths != 0
    mape = np.mean(np.abs(
        (ground_truths[non_zero_indices] - predictions[non_zero_indices]) / ground_truths[non_zero_indices])) * 100

    return mse, rmse, mae, mape


# Helper function to save model checkpoint
def save_model_checkpoint(model, filepath='best_model.pth'):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


# Helper function to load model checkpoint
def load_model_checkpoint(model, filepath='best_model.pth', device='cpu'):
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"Model loaded from {filepath}")
    return model


# Early stopping class to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


# Function to check for NaN or Inf values in tensors
def check_invalid_values(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True
    return False
