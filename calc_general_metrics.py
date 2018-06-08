import matplotlib.pyplot as plt
from sklearn import metrics

from utils import cvt_to_onehot, plot_confusion_matrix


# ======================================
# Prepare Data
Y_test_idx = [1, 0, 1, 0, 1]
Y_pred_idx = [1, 0, 0, 0, 1]

Y_test = cvt_to_onehot(Y_test_idx)
Y_pred = cvt_to_onehot(Y_pred_idx)


# ======================================
# General Metrics


# ======================================
# Calc precision, recall, fscore and support
(precision,
 recall,
 fscore,
 support) = metrics.precision_recall_fscore_support(y_true=Y_test_idx,
                                                    y_pred=Y_pred_idx,
                                                    beta=1.,
                                                    average='binary')

print("precision: {}\nrecall: {}\nfscore: {}\nsupport: {}".format(
        precision, recall, fscore, support))


# ======================================
# Draw Roc curve and Calculate AUC
fpr, tpr, thresholds = metrics.roc_curve(Y_test_idx, Y_pred[:, 1], pos_label=1)
auc_score = metrics.auc(fpr, tpr)

title = 'ROC curve'
fig = plt.figure(title)
ax = fig.add_subplot(111)
ax.plot(fpr, tpr, linestyle='--', lw=2, color='r',
        label='Sentiment Classifier (area = {:.2f})'.format(auc_score))
ax.plot([0., 1.], [0., 1.], linestyle='--', lw=2, color='k',
        label='Random Guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(title)
ax.legend(loc='best', frameon=False)


# ======================================
# Confusion Matrix

cm = metrics.confusion_matrix(Y_test_idx, Y_pred_idx)
classes = ['bad', 'good']

title = 'Confusion matrix'
fig, ax = plot_confusion_matrix(cm,
                                classes=classes,
                                normalize=True,
                                title=title)

