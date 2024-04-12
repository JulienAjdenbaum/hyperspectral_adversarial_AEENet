# Functions from the lab session by Christophe Kervazo on hyperspectral unmixing

import numpy as np

def MultiplicativeUpdate(X, r=None, N_Iter=1000, tolerance=1e-3,
                         A=-1, S=-1, frozenA=False):
    '''
    Inputs: 
    %           X: is a [mxn] matrix to unmig
    %
    %           r: size of the matrices A and S
    %
    %           (Optional) N_Iter: maximum number of iterations
    %
    %           (Optional) tolerance: convergence criteria threshold
    %
    %           (Optional) plot_evolution: plot evolution convergence criteria
    %
    % Outputs:
    %           A: is a [m x n] matrix 
    %           
    %           S: is a [n x t] matrix c
    %
    '''
    if r is None:
        r=X.shape[0]
    
    if frozenA:
        r = A.shape[1]
        
    # Test for positive values
    if np.min(X) < 0:
        raise NameError('Input matrix X has negative values !')      

    # Size
    d,N=X.shape
   
    # Initialization
    if np.any(A < 0) and (not frozenA):
        A=np.random.random((d, r))
    if np.any(S < 0):
        S=np.random.random((r, N)) 
    
    # parameters for convergence
    k = 0
    delta = np.inf
    eps=np.finfo(float).eps
    evolutionDelta=[]
 
    while delta > tolerance and k < N_Iter:
        # Multiplicative method      
        if not frozenA:
            A = A * (X @ S.T) / (A @ S @ S.T + eps) # Add a +eps in the denominator to avoid division by 0
        S = S * (A.T @ X) / (A.T @ A @ S + eps) # Add a +eps in the denominator to avoid division by 0

        # Convergence indices
        k = k + 1           
        diff=X-np.dot(A,S)     

        delta = np.linalg.norm(diff,'fro') / np.linalg.norm(X,'fro') 
        evolutionDelta.append(delta)
        
    return A,S


def VCA(X,r):

    R = X.copy()
    K = np.zeros(r)
    
    for ii in range(r):
        c = np.random.rand(X.shape[0])
        ctX = c.T @ R
        
        p = np.argmax(ctX)
        K[ii] = p
        Rp = np.expand_dims(R[:,p],axis=1)
        
        R = R - (Rp@Rp.T /(np.sum(Rp.squeeze()**2)))@R
        
    print('Max residual %s'%np.max(R))
    
    return K.astype(int)