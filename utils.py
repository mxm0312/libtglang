import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def subset_ind(dataset, ratio):
    return np.random.choice(len(dataset), size=int(ratio*len(dataset)), replace=False)

def plot_train(train_loss, val_loss, val_accuracy):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].set_title('Loss')
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].legend()

    axes[1].set_title('validation accuracy')
    axes[1].plot(val_accuracy)