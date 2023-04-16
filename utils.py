import numpy as np
import prox_tv as ptv
from numpy.linalg import norm
from numpy.random import randn
from sklearn.linear_model import LinearRegression
import time
from tqdm import tqdm
from scipy.linalg import khatri_rao
from numba import jit

'''
--- Version 03.12.2023 ---
Modifications:
    1/ : Remove FISTA, FISTBTLineSearch, use tricks in the naive PGA for acceleration (avoid repeatedly eigendocompistion)
    2/ : Add Nonnegative Initilization using NMF for Nonnegative FL and Nonngative FR

---------
Main funcs:
    KTR_2D(X, M, y, r, lam1, lam2, solver0=1, solver1=1, nonneg = 1, TolFun = 5e-4, Replicates = 10, MaxIter = 125, print_iter = False)
    pga_SFL(y, X, b_init, lam1, lam2, nonneg=1, a = [], delta=1e-4, max_iter=5000, print_iter = False, option=2)
    pga_SFR(y, X, b_init, lam1, lam2, nonneg=1, a = [], delta=1e-4, max_iter=5000, print_iter = False, option=2)
    ALS_TR_2D(X,M,y,r, TolFun=5e-4, Replicates = 10, MaxIter = 125, print_iter = False)

Misc. funcs:
    2. Additive White Gaussian Noise:              AWGN(x_volts, target_snr_db)
---------
'''
''' --- Benchmark 2D: Alternating Least Squares Estimation --- '''
def ALS_TR_2D(X,M,y,r, Replicates = 10, TolFun = 5e-4, MaxIter = 125, print_iter = False):
    """ALS-TR for 2D-variate matrix p1 x p2 regressors and regular vector 
    valued covariates of dimension of dimension p0. By default uses 10 random initial starts, 
    and MaxIter = 125 is the number of iterations. 
    Input:
       X:       n-by-p0 regular covariate matrix
       M:       n-by-p1-by-p2 matrix covariates with dim(M(i,:,:)) = p1 x p2
       y:       n-by-1 respsonse vector
       r:       rank of Kruskal matrix regression
       ifeig:   flag for computing exact eigendecompostion or using trace instead
   Output:
       beta0est:    regression coefficients for the regular covariates
       betaest:     regression coefficients for matrix covariates
       dev:         deviance of final model
       nriters :    the nunmer of iterations before converences for each run (using random start)
    --- 
    """    
    n,p0 = X.shape
    d = len(M.shape)-1            # dimension of array variates
    p = M.shape
    p1, p2 = p[1], p[2]
    dev = float('inf') 
    nriters = np.zeros(Replicates)

    for rep in range(Replicates):
        # ----------------------
        # Initial Start (Random)
        B0 = 1-2*np.random.rand(p1,r)
        B1 = 1-2*np.random.rand(p2,r)

        # Initial Deviance for B = 0 
        LS,_,_,_ = np.linalg.lstsq(X,y,rcond=-1)
        beta0 = LS.reshape(-1,1)
        dev0 = norm(y - X @ beta0)**2
        #print('deviance {:.3f}'.format(dev0))        

        # ------ Iterations Start ------
        for it in range(MaxIter):
            eta0 = X @ beta0
            # -- Update factor matrix B[0] of size p1 x r
            Xj = M @ B1
            Xj = Xj.swapaxes(1,2).reshape(n,p1*r)
            bvec,_,_,_ = np.linalg.lstsq(Xj,y-eta0,rcond=-1)
            B0 = bvec.reshape(r,p1).T

            # -- Update factor matrix B[1] of size p2 x r
            Xj =  M.swapaxes(1,2) @ B0
            Xj = Xj.swapaxes(1,2).reshape(n,p2*r)  
            bvec,_,_,_ = np.linalg.lstsq(Xj,y-eta0,rcond=-1)
            B1 = bvec.reshape(r,p2).T

            # -- Update Reguar Coefficient Vector beta0 of size p0 x 1
            eta = Xj @ bvec.reshape(-1,1) 
            beta0,_,_,_ = np.linalg.lstsq(X,y-eta,rcond=-1)
            beta0 = beta0.reshape(-1,1)
            # -- Check for Convergence
            yhat0 = y - eta - X @ beta0 
            devtmp = norm(yhat0)**2        
            diffdev = devtmp - dev0
            dev0 = devtmp
            abs_err = abs(diffdev)/abs(dev0);
            if print_iter: 
                print("rep {:3d} iter {:3d} abs error {:10.7f} deviance {:.3f}".format(rep,it,abs_err,dev0))
            if (abs_err <TolFun) and (it > 5):
                nriters[rep]=it
                break

        # ------ Iterations Done ------  
        if it == MaxIter:
            print('Max iterations reached in replicate nr',rep)
            nriters[rep]=MaxIter
        # Record if smallest deviance
        if (dev0 < dev) :
            etaest = eta
            beta0est = beta0
            B0est = B0
            B1est = B1
            dev = dev0
            yhat = yhat0
            best_rep = rep
    
        if print_iter: 
            print('replicate: ',rep)
            print(' iterates: ',it)
            print(' deviance: ',dev0)
            print('    beta0: ',beta0)
        
    return beta0est,B0est,B1est,dev,best_rep,nriters

# === General Interface for 2D Tensor/Matrix Regression ===
def KTR_2D(X,M,y,r,lam1,lam2,solver0=1,solver1=1,nonneg=1,\
           B0=[],B1=[],TolFun = 5e-4,Replicates = 10,MaxIter = 125,print_iter = False):
    """ Kruskal Tensor Regression for 2D-variate matrix p1 x p2 regressors and regular vector 
    valued covariates of dimension of dimension p0. By default uses 10 random initial starts, 
    and MaxIter = 125 is the number of iterations. 
    
    Input:
       X            : n-by-p0 regular covariate matrix
       M            : n-by-p1-by-p2 matrix covariates with dim(M(i,:,:)) = p1 x p2
       y            : n-by-1 respsonse vector
       r            : rank of Kruskal matrix regression
       nonneg       : if add nonnegativity as constriant (1 for yes)
       Replicates   : number of random initial starts ( = 10 by default )
       MaxIter      : the number of iterations ( = 125 by default)
       SOLVER handle:
       solver0      :: B0 solver for factor matrix B0 (default: pgaSFL)
       solver1      :: B1 solver for factor matrix B1 (default: pgaSFL)
       
        1 - pga_SFL: Proximal Gradient Algorithm (PGA) for Fused Lasso penalty
            pga_SFL(y,X,b_init,lam1,lam2,nonneg=1,a = [],delta=1e-4,max_iter=5000,print_iter = False)
        -----------------------------------------------------------------------------------------------------------------
        2 - pga_SFR: PGA for Fused Ridge penalty
            pga_SFR(y,X,b_init,lam1,lam2,nonneg=1,a = [],delta=1e-4,max_iter=5000,print_iter = False)
            
   Output:
       beta0est: regression coefficients for the regular covariates
       betaest:  regression coefficients for matrix covariates
       dev: deviance of final model
       nriters : the nunmer of iterations before converences for each run (using random start)
       
    Based on E.O, Feb 6, 2023
    Edited by X.W, Feb 7, 2023
    ---
    2nd version, modified by XJ.W, Mar. 13, 2023:
    Use gradient descent Least Squares...
    """    
    n,p0 = X.shape
    d = len(M.shape)-1            # dimension of array variates
    p = M.shape
    p1, p2 = p[1], p[2]
    dev = float('inf') 
    nriters = np.zeros(Replicates)
    
    if isinstance(B0, list): 
        # random start
        B0init = 1-2*np.random.rand(p1,r)
        B1init = 1-2*np.random.rand(p2,r)
    else:
        # given start 
        B0init = B0
        B1init = B1
    
    for rep in range(Replicates): # remove this Replicates later...

        # Initial start
        B0 = B0init
        B1 = B1init

        # initial deviance for B = 0 
        # LS,_,_,_ = np.linalg.lstsq(X,y,rcond=-1)
        LS,_,_ = gd_LS(y, X, b_init=[], ifeig = 1)
        beta0 = LS.reshape(-1,1)
        dev0 = norm(y - X @ beta0)**2      

        #------ iterations start 
        for it in range(MaxIter):
            eta0 = X @ beta0
            # B0 update factor matrix B[0] of size p1 x r
            Xj = M @ B1
            Xj = Xj.swapaxes(1,2).reshape(n,p1*r)
            if solver0 == 1:
                bvec,_,_ = pga_SFL(y-eta0, Xj, B0.reshape(-1,1), lam1, lam2, nonneg)
            elif solver0 == 2:
                bvec,_,_ = pga_SFR(y-eta0, Xj, B0.reshape(-1,1), lam1, lam2, nonneg)    
            B0 = bvec.reshape(r,p1).T
            
            # B1 update factor matrix B[1] of size p2 x r
            Xj =  M.swapaxes(1,2) @ B0
            Xj = Xj.swapaxes(1,2).reshape(n,p2*r)  
            if solver1 == 1:
                bvec,_,_ = pga_SFL(y-eta0, Xj, B1.reshape(-1,1), lam1, lam2, nonneg)
            elif solver1 == 2:
                bvec,_,_ = pga_SFR(y-eta0, Xj, B1.reshape(-1,1), lam1, lam2, nonneg)
            B1 = bvec.reshape(r,p2).T
            
            #-- update reguar coefficient vector beta0 of size p0 x 1
            eta = Xj @ bvec.reshape(-1,1) 
            # LS,_,_,_ = np.linalg.lstsq(X,y-eta,rcond=-1)
            LS,_,_ = gd_LS(y, X, b_init=[], ifeig = 1)
            beta0 = LS.reshape(-1,1)
            #-- check for convergence 
            yhat0 = y - eta - X @ beta0 
            devtmp = norm(yhat0)**2        
            diffdev = devtmp - dev0
            dev0 = devtmp
            abs_err = abs(diffdev)/abs(dev0);
            if print_iter: 
                print("rep {:3d} iter {:3d} abs error {:10.7f} deviance {:.3f}".format(rep,it,abs_err,dev0))
            if (abs_err <TolFun) and (it > 5):
                nriters[rep]=it
                break

        #------ iterations done 
        if it == MaxIter:
            print('Max iterations reached in replicate nr',rep)
            nriters[rep]=MaxIter
            
        # record if smallest deviance
        if (dev0 < dev) :
            etaest = eta
            beta0est = beta0
            B0est = B0
            B1est = B1
            dev = dev0
            yhat = yhat0
            best_rep = rep
        if print_iter: 
            print('replicate: ',rep)
            print(' iterates: ',it)
            print(' deviance: ',dev0)
            print('    beta0: ',beta0)
    return beta0est,B0est,B1est,dev,best_rep,nriters

def pga_SFL(y, X, b_init, lam1, lam2, nonneg=1, a = [], delta=1e-4, max_iter=5000, print_iter = False):
    '''
    Proximal Gradient Algorithm (PGA) for Fused Lasso penalty
                bvec = pga_SFL(y-eta0, Xj, LS.coef_.reshape(-1,1), lam1, lam2)

    inputs:
        y, X        N x 1 vector of responses and N x p matrix of predictors
        b_init      initial value to start the iteration (p x 1 vector)
        lam1        penalty parameter (non-negative real) for l_1-penalty
        lam2        penalty parameter (non-negative real) for FL-penalty
        a           fixed stepsize (needs to be smaller than 1/L, where L is the max eigenvalue of X^T*X)
        delta       termination threshold
        max_iter    maximum count of iterations    
        print_iter  boolean (True / False) for printing or not printing information. 
    
    Returns: final coefficients, and the optimization history of coefficients for FL
    with no intercept, where the objective function is: 
    
    f(b) = 1/2 * || y - X*b ||^2 + +lambda1 * sum_(i=1)^p | b[i]| +lambda2 * sum_(i=1)^(p-1) | b[i] - b[i+1] | 
    ---
    1st version, Esa Ollila, Dec. 15 2022. 
    --- 
    2nd version, modified by XJ.W, Mar. 13, 2023
    1/1 : Remove option==1 (never being used)
    '''
    # Proximal Operator for L1 norm
    soft = lambda x, lam: np.sign(x)*np.maximum(np.abs(x)-lam,0)   
    # Fused Lasso objective 
    flobj = lambda b, lam1, lam2: 0.5*norm(y-X@b)**2 + \
            lam1 * np.sum(np.abs(b)) + lam2 * np.sum(np.abs(b[1:]-b[:-1]))
    S = []
    # Lipschitz Constant Computation
    if not a:
        # if stepsize not given then compute the step size as 1/2 x max eigenvalue of X^T*X)
        S = X.T@X
        evals = np.linalg.eigvals(S)
        L = np.max(np.real(evals))
        # Fix the S being all zero matrix (Feb.7th, XJW)
        if L < 1e-10: 
            return b_init, [], 1
        a = 1./(2*L)
    # Initialization
    b_old = b_init.copy()
    b_hist = []    
    # This is update using b <- prox(t*c + S*b)
    c  = X.T@y  
    if len(S)==0:
        S = X.T@X
    S  = np.eye(X.shape[1]) - a*(S)
    # Iteration Starts Here
    for i in range(max_iter):
        if lam2 == 0: # LASSO
            b =  soft(a*c + S@b_old, a*lam1)
        else: # Fused LASSO
            b =  soft(ptv.tv1_1d(a*c + S@b_old,a*lam2).reshape(-1,1),a*lam1)
        if nonneg == 1: # Nonnegativity Proximal Opreator
            b =  np.clip(b, a_min=0, a_max=None)
        b_hist.append(b)
        # Stopping Criteria; fixed the problem of “b” being all zeros (Feb.7th, XJW)
        if norm(b) < 1e-10:
            err = 0.5*delta
        else:
            err = norm(b_old-b)/norm(b)
        if print_iter:
            objval = flobj(b,lam1,lam2)
            print("iter {:2d}: error = {:.6f} obj. function = {:.7f}".format(i,err,objval))
        if  err < delta:
            objval = flobj(b,lam1,lam2)
            #print("converged: iteration {:2d}: error = {:.4f} obj. function = {:.7f}".format(i,err,objval))
            return b, b_hist, i
        b_old = b
    return b, b_hist, i

@jit(nopython=True)
def pga_SFR(y, X, b_init, lam1, lam2, nonneg=1, a = [], delta=1e-4, max_iter=5000, print_iter = False, option=2):
    '''
    Proximal Gradient Algorithm (PGA) for Fused Ridge penalty
    bvec = pga_SFR(y-eta0, Xj, LS.coef_.reshape(-1,1), lam1, lam2)

    inputs:
        y, X        N x 1 vector of responses and N x p matrix of predictors
        b_init      initial value to start the iteration (p x 1 vector)
        lam1        penalty parameter (non-negative real) for l_1-penalty
        lam2        penalty parameter (non-negative real) for FL-penalty
        a           fixed stepsize (needs to be smaller than 1/L, where L is the max eigenvalue of X^T*X)
        delta       termination threshold
        max_iter    maximum count of iterations    
        print_iter  boolean (True / False) for printing or not printing information. 
    
    Returns: final coefficients, and the optimization history of coefficients for FL
    with no intercept, where the objective function is: 
    
    f(b) = 1/2 * || y - X*b ||^2 + +lambda1 * sum_(i=1)^p | b[i]| +lambda2 * sum_(i=1)^(p-1) | b[i] - b[i+1] | 
    ---
    1st version, Esa Ollila, Dec. 15 2022. 
    --- 
    2nd version, modified by XJ.W, Mar. 13, 2023
    1/1 : Remove option==1 (never being used)
    '''
    prox_l2 = lambda x, lam: (1 - lam/(np.maximum(norm(x),lam)))*x 
    # http://proximity-operator.net/multivariatefunctions.html
    # Fused Ridge objective 
    frobj = lambda b, lam1, lam2: 0.5*norm(y-X@b)**2 + \
            lam1 * np.sum(np.square(b)) + lam2 * np.sum(np.square(b[1:]-b[:-1]))
    S = []
    # Lipschitz Constant Computation
    if not a:
        # if stepsize not given then compute the step size as 1/2 x max eigenvalue of X^T*X)
        S = X.T@X
        evals = np.linalg.eigvals(S)
        L = np.max(np.real(evals))
        # Fix the S being all zero matrix (Feb.7th, XJW)
        if L < 1e-10: 
            return b_init, [], 1
        a = 1./(2*L)
    # Initialization
    b_old = b_init.copy()
    b_hist = []    
    # This is update using b <- prox(t*c + S*b)
    c  = X.T@y  
    if len(S)==0:
        S = X.T@X
    S  = np.eye(X.shape[1]) - a*(S)
    # Iteration Starts Here
    for i in range(max_iter):
        if lam2 == 0: # LASSO
            b =  prox_l2(a*c + S@b_old, a*lam1)
        else: # Fused LASSO
            b =  prox_l2(ptv.tv2_1d(a*c + S@b_old,a*lam2).reshape(-1,1),a*lam1)
        if nonneg == 1: # Nonnegativity Proximal Opreator
            b =  np.clip(b, a_min=0, a_max=None)
        b_hist.append(b)
        # Stopping Criteria; fixed the problem of “b” being all zeros (Feb.7th, XJW)
        if norm(b) < 1e-10:
            err = 0.5*delta
        else:
            err = norm(b_old-b)/norm(b)
        if print_iter:
            objval = frobj(b,lam1,lam2)
            print("iter {:2d}: error = {:.6f} obj. function = {:.7f}".format(i,err,objval))
        if  err < delta:
            objval = frobj(b,lam1,lam2)
            #print("converged: iteration {:2d}: error = {:.4f} obj. function = {:.7f}".format(i,err,objval))
            return b, b_hist, i
        b_old = b

    return b, b_hist, i

''' --- Gradient Descent for LS ---'''
# @jit(nopython=True)
def gd_LS(y, X, b_init=[], nonneg = 0, ifeig = 1, delta=1e-4, max_iter=5000, print_iter = False):
    '''
    Gradient Descent for Least Square (approximation)
    inputs:
        y, X        N x 1 vector of responses and N x p matrix of predictors
        b_init      initial value to start the iteration (p x 1 vector)
        nonneg      nonnegative LS estimation, only for providing a nonneg start for nnFL, nnFR...
        ifeig       flag for computing exact eigendecompostion or using trace instead
        delta       termination threshold
        max_iter    maximum count of iterations    
        print_iter  boolean (True / False) for printing or not printing information. 
    
    f(b) = 1/2 * || y - X*b ||^2 
    --- 
    1st version, modified by XJ.W, Mar. 16, 2023
    '''
    flsobj = lambda b: 0.5*norm(y-X@b)**2  # Least Square objective 
    # Lipschitz Constant Computation
    S = X.T@X
    c  = X.T@y
    if ifeig == 1:
        evals = np.linalg.eigvals(S)
        a = 1./np.max(np.real(evals))
    else:
        a = 1./np.trace(S)
    if len(b_init) == 0:
        b_init = np.random.randn(*c.shape)
    # Initialization
    b_old = b_init.copy()
    b_hist = []    
    # This is update using b <- prox(t*c + S*b)
    S  = np.eye(X.shape[1]) - a*(S)
    # Iteration Starts Here
    for i in range(max_iter):
        b = a*c + S@b_old
        if nonneg == 1: # Nonnegativity Proximal Opreator
            b =  np.clip(b, a_min=0, a_max=None)
        b_hist.append(b)
        # Stopping Criteria; fixed the problem of “b” being all zeros (Feb.7th, XJW)
        if norm(b) < 1e-10:
            err = 0.5*delta
        else:
            err = norm(b_old-b)/norm(b)
        if print_iter:
            objval = flsobj(b)
            print("iter {:2d}: error = {:.6f} obj. function = {:.7f}".format(i,err,objval))
        if  err < delta:
            objval = flsobj(b)
            return b, b_hist, i
        b_old = b
    return b, b_hist, i


# --- Miscellaneous Functions --- 
''' 
Function: Adding noise using target SNR 
'''
# @jit(nopython=True)
def AWGN(x_volts, target_snr_db):
    '''
    Add white Gaussian noise to a pure signal
    
    Parameters
    ----------
    x_volts: pure signal
    target_snr_db: Target SNR / dB
    
    Returns
    -------
    y_volts: signal added with white Gaussian noise with Target SNR!
    
    '''
    x_volts = x_volts.squeeze()
    x_watts = x_volts ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise_volts = np.random.normal(0, np.sqrt(noise_avg_watts), len(x_watts))
    y_volts = x_volts + noise_volts
    return y_volts