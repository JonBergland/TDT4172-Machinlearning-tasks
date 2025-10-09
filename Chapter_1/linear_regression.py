import numpy as np
import pandas as pd

class LinearRegression():
    
    def __init__(self, epochs = 30000, learning_rate = 0.002):
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.losses, self.train_accuracies = [], []

        pass
        
    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            lin_model = np.dot(x, self.weights) + self.bias

            y_pred = lin_model
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)


            loss = self.compute_loss(y, y_pred) 
            self.losses.append(loss)


    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def compute_gradients(self, x, y, y_pred):
        m = x.shape[0]
        error = y_pred - y
  
        grad_w = np.dot(x.T, error) / m
        grad_b = np.mean(error)

        return grad_w, grad_b
        

    def compute_loss(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)


    
    def predict(self, x: pd.DataFrame):
        lin_model = np.dot(x, self.weights) + self.bias
        return lin_model