"""
functions to compute structural impulse responses give VAR(p) parameter
estimates

@author: koehnden
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_struc_irf(df,Theta,shock_in,horizon=10):
    j = df.columns.get_loc(shock_in)
    irf = np.zeros((horizon+1,1))
    plt.figure(1,figsize=(20,10))
    plt.suptitle("IRF for shocks in " + shock_in, fontsize=30)
    for i in range(0,df.shape[1]):
        for h in range(0,horizon):
            irf[h+1] = Theta[:][h][j][i]
        plt.subplot(221 + i)
        plt.title("Effect on " + df.columns[i] + " given shock in " + shock_in)
        plt.plot(range(0,horizon+1), irf, color='b', linewidth=2.0)

def get_struct_ir(coefs,B_0,last_horizon):
    p, k, k2 = coefs.shape
    assert(k == k2)
    A_companion = get_companion_matrix(coefs,p,k) 
    Theta = np.zeros((last_horizon,k,k))
    for i in range(0,last_horizon):
        Phi_i = get_reduced_form_ir(A_companion,p,k,i)
        if (i == 0):
            Theta_i = np.linalg.inv(B_0)
        else:
            Theta_i = np.dot(Phi_i,np.linalg.inv(B_0))
        Theta[i] = Theta_i
    return Theta


def get_reduced_form_ir(A_companion,p,k,horizon):
    """
    Return Reduce form impuls response
    Phi_i = J A^i J where i is the horizon and J := [I_k, 0_kxk(p-1)]
    """
    identity_k = np.identity(k)
    zeros_k_pk = np.zeros((k,k*(p-1)))
    J = np.concatenate((identity_k, zeros_k_pk), axis=1)
    Phi = np.dot(J,np.dot(A_companion**horizon,J.T))
    return Phi
    
def get_companion_matrix(coefs, k, p):
    """
    Return compansion matrix for the VAR(1) representation for a VAR(p) process
    (companion form)
    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    p, k, k2 = coefs.shape
    assert(k == k2)

    kp = k * p

    result = np.zeros((kp, kp))
    result[:k] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k, kp), np.arange(kp-k)] = 1

    return result