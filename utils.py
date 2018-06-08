import itertools

import numpy as np
import matplotlib.pyplot as plt


def cvt_to_onehot(Y, dim):
    Y_onehot = np.zeros((len(Y), dim), dtype=np.float32)
    Y_onehot[np.arange(len(Y)), Y] = 1.0
    return Y_onehot


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure(title)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return fig, ax



