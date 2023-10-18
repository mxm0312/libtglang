import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

HIDDEN_CHANNEL = 4

class CodeSnippetClf(torch.nn.Module):
    # Defining the Constructor
    def __init__(self, alphabet_size):
        super(CodeSnippetClf, self).__init__()
        # Conv layers
        self.conv1 = torch.nn.Conv1d(alphabet_size, HIDDEN_CHANNEL * 16, 3, stride=3)
        self.conv2 = torch.nn.Conv1d(HIDDEN_CHANNEL * 16, HIDDEN_CHANNEL * 8, 3, stride=3)
        self.conv3 = torch.nn.Conv1d(HIDDEN_CHANNEL * 8, HIDDEN_CHANNEL * 2, 3, stride=3)
        # Pooling
        self.pool1 = torch.nn.MaxPool1d(2, stride=2)
        # Regularization
        self.dropout = torch.nn.Dropout(0.2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(8 * HIDDEN_CHANNEL, 16 * HIDDEN_CHANNEL)
        self.fc2 = torch.nn.Linear(16 * HIDDEN_CHANNEL, 81)
        self.fc3 = torch.nn.Linear(81, 100)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x))) 
        x = F.relu(self.pool1(self.conv2(x))) 
        x = F.relu(self.pool1(self.conv3(x)))  
        # Flatten
        x = x.view(-1, HIDDEN_CHANNEL * 8)
        # Feed to fully-connected layer to predict class
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # Return class probabilities via a log_softmax function 
        return self.activation(x)
    


def train_model(model, train_loader, val_loader, loss_f, opt, epochs):
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(epochs):
        ep_train_loss = []
        ep_val_loss = []
        ep_val_accuracy = []
        start_time = time.time()
       
        model.train(True) # enable dropout / batch norm
        for X_batch, Y_batch in train_loader:
            predictions = model(X_batch)
            opt.zero_grad()

            loss = loss_f(predictions, Y_batch)
            loss.backward()
            opt.step()
            ep_train_loss.append(loss.item())
            
        torch.save(model, f'code_snippet_clf_{epoch}.pht')
        model.train(False)
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                predictions = model(X_batch)

                loss = loss_f(predictions, Y_batch)
                ep_val_loss.append(loss.item())
                
                ep_val_accuracy.append(np.mean( (torch.argmax(predictions, dim=1) == torch.argmax(Y_batch, dim=1) ).numpy() ))

        print(f'Epoch {epoch + 1}/{epochs}. time: {time.time() - start_time:.3f}s')

        train_loss.append(np.mean(ep_train_loss))
        val_loss.append(np.mean(ep_val_loss))
        val_accuracy.append(np.mean(ep_val_accuracy))

        print(f'train loss: {train_loss[-1]:.6f}')
        print(f'val loss: {val_loss[-1]:.6f}')
        print(f'validation acc: {val_accuracy[-1]:.6f}')
        print('\n')

    return train_loss, val_loss, val_accuracy