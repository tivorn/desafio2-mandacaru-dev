from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(real, predicted, classes):
    cm = confusion_matrix(real, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
                cbar=False)
    ax.set(xlabel="Predito", ylabel="Real", xticklabels=classes, 
        yticklabels=classes, title="Matriz de confusão")
    plt.yticks(rotation=0)

def plot_roc_curve(y, y_prob, classes):
    for i in range(len(classes)):
        fpr, tpr, thresholds = roc_curve(y[:, i], y_prob[:, i])
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label=f'{classes[i]} (Área={round(area, 3)})')
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.xlim(-0.05, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('Taxa de falso positivo')
        plt.ylabel('Taxa de verdadeiro positivo')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True)

def plot_precision_recall_curve(y, y_prob, classes):
    for i in range(len(classes)):
        precision, recall, thresholds = precision_recall_curve(y[:, i], y_prob[:, i])
        area = auc(recall, precision)
        plt.plot(recall, precision, lw=3, label=f'{classes[i]} (Área={round(area, 3)})')
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.xlim(-0.05, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('Revocação')
        plt.ylabel('Precisão')
        plt.title('Curva Precisão-Revocação')
        plt.legend(loc="lower right")
        plt.grid(True)