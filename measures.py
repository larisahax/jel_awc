import numpy as np
import math


def get_error(weights, true_weights, separate_errors = False):
    '''Counts the propagation and separation errors, also general error
    
    Args:
        weights (ndarray): weight matrix resulted from AWC
        true_weights (ndarray): true weight matrix
        separate_errors (bool, optional): If True it will return propagation and separation errors, otherwise the general error will be counted. By default False.  
     
     Returns:
        float, float: The propagation and separation errors
        or
        float: the general error. 
    '''
    error_1 = np.sum(np.abs(weights) * (true_weights == 0)) / np.sum(np.abs(1 - true_weights) * (true_weights == 0))
    error_2 = np.sum(np.abs(weights - 1) * (true_weights == 1)) / np.sum(np.abs(np.identity(np.size(weights,0)) - 1) * (true_weights == 1))
        
    if separate_errors:
        return error_1, error_2
    else:
        return (np.sum(np.abs(weights) * (true_weights == 0)) + np.sum(np.abs(weights - 1) * (true_weights == 1))) *1. / np.size(weights,0) / (np.size(weights,0)-1)


def Fscore(answer_clusters, true_clusters,  n):
    '''Compute the F score between answer and true clusters'''
    n *= 1.
    F = np.zeros((len(true_clusters), len(answer_clusters)))
    
    F_score = 0.
    
    for i in xrange(len(true_clusters)):
        c_1 = true_clusters[i]
        for j in xrange(len(answer_clusters)):
            c_2 = answer_clusters[j]
            n_ij = 1. * len(set(c_1).intersection(set(c_2)))
            n_i = len(c_1) * 1.
            n_j = len(c_2) * 1.
            if n_ij != 0:
                F[i, j] = (2. * n_ij) / (n_i + n_j)
        F_score += n_i  * np.max(F[i, :])
    F_score /= n
    return max(F_score, 0)

def NMI(clusters_1, clusters_2, n):
    '''Compute the Normilized Mutual Information (NMI) between 2 cluster structures'''
    from sklearn.metrics.cluster import normalized_mutual_info_score
    Y_1 = np.zeros((n,))
    Y_2 = np.zeros((n,))

    for i in xrange(len(clusters_1)):
        Y_1[clusters_1[i]] = i
    for i in xrange(len(clusters_2)):
        Y_2[clusters_2[i]] = i
    return max(normalized_mutual_info_score(Y_1, Y_2), 0)

def AdRand(answer_clusters, true_clusters, n):
    from sklearn.metrics.cluster import adjusted_rand_score
    Y_1 = np.zeros((n,))
    Y_2 = np.zeros((n,))
    for i in xrange(len(answer_clusters)):
        Y_1[answer_clusters[i]] = i
    for i in xrange(len(true_clusters)):
        Y_2[true_clusters[i]] = i
    
    return max(adjusted_rand_score(Y_2, Y_1), 0)

def Purity(answer_clusters, true_clusters,  n):
    '''Compute the Purity between answer and true clusters'''
    n *= 1.
    n_ij = np.zeros((len(true_clusters), len(answer_clusters)))
    Purity = 0.
    for i in xrange(len(true_clusters)):
        c_1 = true_clusters[i]
        for j in xrange(len(answer_clusters)):
            c_2 = answer_clusters[j]
            n_ij[i,j] = 1. * len(set(c_1).intersection(set(c_2)))
        Purity += np.max(n_ij[i, :])
    Purity /= n
    return Purity

def Entropy(answer_clusters, true_clusters,  n):
    '''Compute the Entropy between answer and true clusters'''
    n *= 1.
    n_ij = np.zeros((len(true_clusters), len(answer_clusters)))
    
    Entropy = 0.
    
    for i in xrange(len(true_clusters)):
        c_1 = true_clusters[i]
        n_i = len(c_1) * 1.
        for j in xrange(len(answer_clusters)):
            c_2 = answer_clusters[j]
            n_ij[i,j] = 1. * len(set(c_1).intersection(set(c_2)))
            n_j = len(c_2) * 1.
            if n_ij[i,j] != 0:
                Entropy -= n_ij[i,j] * np.log(n_ij[i,j] / n_i)
    Entropy /= n
    Entropy /= math.log(1. * len(true_clusters))
    return Entropy


def correct_NMI(true, answer, n, score):
    Y_1 = np.zeros((n,))
    Y_2 = np.zeros((n,))
    for i in xrange(len(true)):
        Y_1[true[i]] = i
    for i in xrange(len(answer)):
        Y_2[answer[i]] = i
    true = Y_1
    answer = Y_2
    true_clusters = np.unique(true)
    answer_clusters = np.unique(answer)
    M = len(true_clusters)
    K = len(answer_clusters)
    true_dict = {}
    for i, cl in enumerate(true_clusters):
        true_dict[cl] = i
    answer_dict = {}
    for i, cl in enumerate(answer_clusters):
        answer_dict[cl] = i
    #print true_dict
    n = np.zeros((M, K))
    
    N = len(true)
    for i in xrange(N):
        n[true_dict[true[i]], answer_dict[answer[i]]] += 1.

    n_true   = np.sum(n, axis=1)
    n_answer = np.sum(n, axis=0)
    
    #N = np.sum(n)
    
    #print n
    if score == 'nmi':
        MI = 0
        for m in xrange(M):
            for k in xrange(K):
                MI += n[m, k] * math.log(max(n[m,k] * N / n_true[m] / n_answer[k], 0.0000000000001))
                #print m,k,n[m, k], n_true[m] * n_answer[k], n_true[m], n_answer[k]
        H_true = 0
        for m in xrange(M):
            H_true += n_true[m] * math.log(max(n_true[m] / N, 0.0000000000001))
        H_answer = 0
        for k in xrange(K):
            H_answer += n_answer[k] * math.log(max(n_answer[k] / N, 0.0000000000001))
        NMI = - 2*MI / (H_true + H_answer)
        return NMI
    
    nn = np.zeros((M, K))
    
    N = len(true)
    for i in xrange(N):
        for k in xrange(K):
            nn[true_dict[true[i]], k] += 1.
        for m in xrange(M):
            nn[m, answer_dict[answer[i]]] += 1.
    
    J = n / (nn - n)
    
    #print nn
    if score == 'max_recall':
        recall = np.max(J, axis=1)
        return recall
    if score == 'avg_recall':
        recall = np.max(J, axis=1)
        return np.average(recall)
    if score == 'max_precision':
        precision = np.max(J, axis=0)
        return precision
    if score == 'avg_precision':
        precision = np.max(J, axis=0)
        return np.average(precision)
    

