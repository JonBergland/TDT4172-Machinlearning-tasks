import numpy as np

class LinearRegression():
    
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.epochs = 5
        self.weights = [0]
        self.bias = 0
        self.learning_rate = 0.1
        self.losses, self.train_accuracies = [], []

        pass
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement

        #Init
        self.weights = np.zeros(X.shape[0])
        self.bias = 0

        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias

            y_pred = lin_model
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)


            loss = self.compute_loss(y, y_pred) 
            # pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred] 
            # self.train_accuracies.append(accuracy(y, pred_to_class)) 
            self.losses.append(loss)




        # raise NotImplementedError("The fit method is not implemented yet.")


    def update_parameters(self, grad_w, grad_b):
        self.weights += grad_w
        self.bias += grad_b

    def compute_gradients(self, x, y, y_pred):
        grad_w = []
        for i in (x.transpose()):
            grad_w.append((y_pred-y)*x[i])

        grad_b = y_pred - y

        return grad_w, grad_b
        

    def compute_loss(self, y, y_pred):
        return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)


    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        lin_model = np.matmul(X, self.weights) + self.bias 
        # y_pred = self._sigmoid(lin_model) 
        y_pred = lin_model
        return [1 if _y > 0.5 else 0 for _y in y_pred]
        # raise NotImplementedError("The predict method is not implemented yet.")





