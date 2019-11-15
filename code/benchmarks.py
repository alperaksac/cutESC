import json

import jqmcvi.base as jqmcvin
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix


def confusion_matrix(labels_true, labels_pred):
    n_samples = len(labels_true)
    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    total = n_samples * (n_samples-1)
    tp = np.dot(c.data, c.data) - n_samples
    fp = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples - tp
    fn = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples - tp
    tn = total - (tp+fn+fp)
    return tp, tn, fp, fn


def dice_score(tp, fp, fn):
    return (2*tp)/(2*tp+fp+fn)


def jaccard_score(tp, fp, fn):
    return tp/(tp+fp+fn)


def precision_score(tp, fp):
    return tp/(tp+fp)


def recall_score(tp, fn):
    return tp/(tp+fn)


def benchmarks(data, labels_true, labels_pred, verbose=False):
    tp, tn, fp, fn = confusion_matrix(labels_true, labels_pred)
    results = {}

    results['Silhouette Coefficient'] = round(metrics.silhouette_score(
        data, labels_pred), 3)
    results['Calinski Harabaz'] = round(metrics.calinski_harabasz_score(
        data, labels_pred))
    results['Dunn Index'] = round(jqmcvin.dunn_fast(data, labels_pred), 3)

    results['Homogeneity'] = round(metrics.homogeneity_score(
        labels_true, labels_pred), 3)
    results['Completeness'] = round(metrics.completeness_score(
        labels_true, labels_pred), 3)
    results['V-Measure'] = round(
        metrics.v_measure_score(labels_true, labels_pred), 3)

    results['Dice'] = round(dice_score(tp, fp, fn), 3)
    results['Jaccard'] = round(jaccard_score(tp, fp, fn), 3)
    results['Precision'] = round(precision_score(tp, fp), 3)
    results['Recall'] = round(recall_score(tp, fn), 3)

    results['Fowlkes Mallows'] = round(metrics.fowlkes_mallows_score(
        labels_true, labels_pred), 3)
    results['Adjusted Rand Index'] = round(metrics.adjusted_rand_score(
        labels_true, labels_pred), 3)
    results['Adjusted Mutual Information'] = round(metrics.adjusted_mutual_info_score(
        labels_true, labels_pred, average_method='max'), 3)

    if verbose:
        print(80 * '*')
        print('Silhouette Coefficient\tDunn Index\tCalinski Harabaz')
        print('%.3f\t\t\t%.3f\t\t\t%.3f'
              % (results['Silhouette Coefficient'],
                 results['Dunn Index'],
                 results['Calinski Harabaz']))
        print('Dice\tJaccard\tPrecision\tRecall')
        print('%.3f\t%.3f\t%.3f\t\t%.3f'
              % (results['Dice'],
                 results['Jaccard'],
                 results['Precision'],
                 results['Recall']))
        print('Fowlkes Mallows\tAdjusted Rand Index\tAdjusted Mutual Information')
        print('%.3f\t\t%.3f\t\t\t%.3f'
              % (results['Fowlkes Mallows'],
                 results['Adjusted Rand Index'],
                 results['Adjusted Mutual Information']))
        print('Homogeneity\tCompleteness\tV-measure')
        print('%.3f\t\t%.3f\t\t%.3f'
              % (results['Homogeneity'],
                 results['Completeness'],
                 results['V-Measure']))
        print(80 * '*')
    return results


def draw_clusters(X, labels_pred):
    plt.figure()
    unique_labels = set(labels_pred)
    colors = np.random.rand(len(unique_labels), 3)
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels_pred == k)
        xy = X[class_member_mask]
        size = 4
        if k == -1:
            col = [0, 0, 0, 1]
            size = 2
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(
            col), markeredgewidth=0, markersize=size)
    plt.axis('off')
    return plt


def draw_network(G, X, labels_pred):
    plt.figure()
    unique_labels = set(labels_pred)
    colors = np.random.rand(len(unique_labels), 3)
    colors = [colors[labels_pred[n]] for n in G.nodes()]
    ec = nx.draw_networkx_edges(G, pos=X, alpha=0.5, width=0.5)
    nc = nx.draw_networkx_nodes(G, pos=X, node_size=5,
                                node_color=colors)
    plt.axis('off')
    return plt


def write_plot(name, plt):
    plt.savefig(name, dpi=300)
    plt.close()


def write_results(name, results):
    f = open(name+'_benchs.txt', 'a')
    s = json.dumps(results)
    f.write(s+'\n')
    f.close()
