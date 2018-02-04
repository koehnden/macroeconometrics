"""
script that provides function to impose sign restrictions 

@author: koehnden
"""
import numpy as np


def satisfies_restrictions(candidate, restrictions):
    """
    Returns True if all restriction specified in restriction are satisfied
    
    input:
        restrictions: a numpy array in which each element is a list that
                      contains a tupel (row,col) of the position of the sign restrictions 
                      in B_0 and the sign of the restriction -1 or 1
    """
    
    restrictions_fulfilled = True
    for restriction in restrictions:
        i, j = restriction[0]
        element_to_check = candidate[i][j]
        if (np.sign(element_to_check) != np.sign(restriction[1])):
            restrictions_fulfilled = False
    return restrictions_fulfilled

def qr_with_positive_diagonals(W):
    """
    convert the corresponding elements of Q if the diagonal
    of R is negative
    """
    Q, R  = np.linalg.qr(W)
    R_diag = np.diagonal(R)
    for i in range(0,Q.shape[0]):
        if (R_diag[i] < 0):
            Q[i][i] *= -1
    return Q


def generate_gaussian_matrix(k):
    """
    Returns a matrix W whose row are generated by w~N(0,I_k)
    """
    mu, sigma = 0, 1
    W = np.empty((k,k))
    for i in range(0,k):
        W[i] = np.random.normal(mu, sigma, k)
    return W