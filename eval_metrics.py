import numpy as np
from scipy import interpolate
from sklearn.metrics import auc
from sklearn.model_selection import KFold


def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics

    thresholds = np.arange(0, 5, 0.01)
    tpr, fpr, precision, recall, accuracy, best_distances = calculate_roc(thresholds, distances,
                                       labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 5, 0.001)
    tar, far = calculate_val(thresholds, distances,
                                      labels, 1e-3, nrof_folds=nrof_folds)
    roc_auc = auc(fpr, tpr)
    
    return tpr, fpr, precision, recall, accuracy, roc_auc, tar, far, best_distances


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    precision = np.zeros((nrof_folds))
    recall = np.zeros((nrof_folds))
    accuracy = np.zeros((nrof_folds))
    best_distances = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)


        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, _, _ = calculate_accuracy(threshold,
                                                                                                 distances[test_set],
                                                                                                 labels[test_set])
        _, _, precision[fold_idx], recall[fold_idx], accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set],
                                                      labels[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        best_distances[fold_idx] = thresholds[best_threshold_index]
        best_distance = np.mean(best_distances, 0)
    return tpr, fpr, precision, recall, accuracy, best_distance


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))


    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    precision = 0 if (tp + fp) == 0 else float(tp) / float(tp + fp)
    recall = 0 if (tp + fn) == 0 else float(tp) / float(tp + fn)

    acc = float(tp + tn) / dist.size
    return tpr, fpr, precision, recall, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tar = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_tar_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_idx], far[fold_idx] = calculate_tar_far(threshold, distances[test_set], labels[test_set])

    tar_mean = np.mean(tar)
    far_mean = np.mean(far)
    # val_std = np.std(val)
    return tar_mean, far_mean


def calculate_tar_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    tar = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return tar, far


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(15,15))
    lw = 2
    plt.plot(fpr, tpr, color='#16a085',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#2c3e50', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", frameon=False)
    fig.savefig(figure_name, dpi=fig.dpi)
    plt.close()
