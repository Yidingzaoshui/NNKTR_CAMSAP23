import numpy as np
import jax
import jax.numpy as jnp
import prox_tv as ptv

@jax.jit
def cg_LS_lax(X, y):
    ''' Conjugate Gradient Method for Least Square: 0.5 * ||y - Xb||^2 
        Inputs: 
            X: \in (m x n)
            y: \in (m)
        Outputs: 
            b0: \in (n)
        Written By WXJ, 0327-2023 '''
    lsobj = lambda b: 0.5*np.linalg.norm(y-X@b)**2
    max_iter, delta = int(1e5), 1e-4 
    key = jax.random.PRNGKey(0) # Setting random seed
    b0 = jax.random.uniform(key, shape=(X.T@y).shape)
    d0 = y - X @ b0
    r0 = X.T @ d0
    p0 = r0.copy()
    t0 = X @ p0
    def cond_fun(inputs):
        i, err, *_ = inputs
        return (i < max_iter) & (err >= delta)
    def body_fun(inputs): 
        i, err, b0,d0,r0,p0,t0 = inputs
        alpha = (jnp.linalg.norm(r0)/jnp.linalg.norm(t0))**2
        b1 = b0 + alpha*p0    
        err = jnp.linalg.norm(b1-b0)/jnp.linalg.norm(b1) # Stopping Criteria
        d1 = d0 - alpha*t0
        r1 = X.T @ d1
        p1 = r1 + ((jnp.linalg.norm(r1)/jnp.linalg.norm(r0))**2)*p0
        t1 = X @ p1
        r0,t0,p0,d0,b0 = r1.copy(), t1.copy(),p1.copy(),d1.copy(),b1.copy()
        return (i+1, err, b0,d0,r0,p0,t0)
    i, err = 0, 1.0
    inputs = (i, err, b0,d0,r0,p0,t0)  # Initializing
    i,err, b0,d0,r0,p0,t0 = jax.lax.while_loop(cond_fun, body_fun, inputs)
    return b0

@jax.jit
def grad_ls_jit(a,c,S,b_old):
    return a*c+S@b_old

def soft(x, lam1): # Proximal Operator of L1-norm
    return np.sign(x) * np.maximum(np.abs(x) - lam1, 0)

# ------------- Fused Lasso ------------- 
def pga_FL_jax(X,y,b_init,lams,nonneg=0,verbose=1):
    ''' Proximal Gradient Algorithm for Fused Lasso
    0.5 * || y - Xb ||^2 + lambda1*sum_(i=1)^p |b[i]| +lambda2*sum_(i=1)^(p-1) | b[i] - b[i+1] | 
    Inputs:
        y, X        (m), (m x n)
        b_init      (n) warm start '''
    IfNonNeg = [-1e10, 0] # Avoid Boolean Expression
    def flobj(y,X,b,lam1,lam2):
        funcvalue = 0.5*jnp.linalg.norm(y-X@b)**2+lam1*jnp.sum(jnp.abs(b))+lam2*jnp.sum(jnp.abs(b[1:]-b[:-1]))
        return funcvalue
    def soft(x, lam1):
        return jnp.sign(x)*jnp.maximum(jnp.abs(x)-lam1,0)
    key = jax.random.PRNGKey(0) # Setting Random Seed
    max_iter, delta = int(1e5), 1e-4 # No need to put this in the arguments
    S, c = X.T @ X, X.T @ y # Precomputing (1/2)
    lam1, lam2 = lams[0], lams[1]
    a = 1.0 / jnp.max(jnp.real(jnp.linalg.eigvals(X.T @ X))) # Lipschitz Constant (2/2)
    b = jax.random.uniform(key, shape=c.shape)
    b_old = jnp.asarray(b_init)
    S = jnp.eye(X.shape[1]) - a * S
    def cond_fun(inputs):
        i, err, *_= inputs
        return (i < max_iter) & (err >= delta)
    def body_fun(inputs):
        i, err, b_old, b = inputs
        # Gradient 
        b =  jnp.clip(soft(ptv.tv1_1d(grad_ls_jit(a,c,S,b_old),a*lam2),a*lam1),a_min=IfNonNeg[nonneg],a_max=None) 
        b = b.reshape(-1,1)
        err = jnp.linalg.norm(b_old - b) / jnp.linalg.norm(b)
        b_old = b
        return (i+1, err, b_old, b)
    i, err = 0, 1.0
    inputs = (i, err, b_old, b) # Initializing
    while cond_fun(inputs):
        (i, err, b_old, b ) = body_fun(inputs) # Main function
        i += 1
        inputs = (i, err, b_old, b)
        if verbose == 1:
            objval = flobj(y,X,b,lam1,lam2)
            print("iter {:2d}: error = {:.7f} obj. function = {:.7f}".format(i,err,objval))
    return b

# ------------- Fused Ridge ------------- 
def pga_FR_jax(X,y,b_init,lams,nonneg=0,verbose=1):
    ''' Proximal Gradient Algorithm for Fused Ridge
    0.5 * || y - Xb ||^2 + lambda1* ||b||_2 +lambda2*sum_(i=1)^(p-1) || b[i] - b[i+1] ||
    Inputs:
        y, X        (m), (m x n)
        b_init      (n) warm start '''
    IfNonNeg = [-1e10, 0] # Avoid Boolean Expression
    def frobj(y,X,b,lam1,lam2):
        funcvalue = 0.5*jnp.linalg.norm(y-X@b)**2+lam1*jnp.linalg.norm(b)+lam2*jnp.sum(jnp.linalg.norm(b[1:]-b[:-1]))
        return funcvalue
    def prox_l2(x, lam1):
        return (1 - lam1/(jnp.maximum(jnp.linalg.norm(x),lam1)))*x 
    key = jax.random.PRNGKey(0) # Setting Random Seed
    max_iter, delta = int(1e5), 1e-4 # No need to put this in the arguments
    S, c = X.T @ X, X.T @ y # Precomputing (1/2)
    lam1, lam2 = lams[0], lams[1]
    a = 1.0 / jnp.max(jnp.real(jnp.linalg.eigvals(X.T @ X))) # Lipschitz Constant (2/2)
    b = jax.random.uniform(key, shape=c.shape)
    b_old = jnp.asarray(b_init)
    S = jnp.eye(X.shape[1]) - a * S
    def cond_fun(inputs):
        i, err, *_= inputs
        return (i < max_iter) & (err >= delta)
    def body_fun(inputs):
        i, err, b_old, b = inputs
        # Gradient 
        b =  jnp.clip(prox_l2(ptv.tv2_1d(grad_ls_jit(a,c,S,b_old),a*lam2),a*lam1),a_min=IfNonNeg[nonneg],a_max=None) 
        b = b.reshape(-1,1)
        err = jnp.linalg.norm(b_old - b) / jnp.linalg.norm(b)
        b_old = b
        return (i+1, err, b_old, b)
    i, err = 0, 1.0
    inputs = (i, err, b_old, b) # Initializing
    while cond_fun(inputs):
        (i, err, b_old, b ) = body_fun(inputs) # Main function
        i += 1
        inputs = (i, err, b_old, b)
        if verbose == 1:
            objval = frobj(y,X,b,lam1,lam2)
            print("iter {:2d}: error = {:.7f} obj. function = {:.7f}".format(i,err,objval))
    return b

def ALS_cg(X,M,y,r, Replicates = 10, MaxIter = 125, verbose = False):
    
    n,p0 = X.shape
    p = M.shape
    p1, p2, dev, TolFun = p[1], p[2], float('inf'), 5e-4
    nriters = np.zeros(Replicates)
    dev_list = np.zeros((Replicates, MaxIter))

    for rep in range(Replicates):
        # Initial Start (Random)
        B0 = 1-2*np.random.rand(p1,r)
        B1 = 1-2*np.random.rand(p2,r)
        beta0 = cg_LS_lax(X,y).reshape(-1,1)
        dev0 = np.linalg.norm(y - X @ beta0)**2

        for it in range(MaxIter):
            eta0 = X @ beta0
            # --- B[0]  (p1,r) --- 
            Xj = M @ B1
            Xj = Xj.swapaxes(1,2).reshape(n,p1*r)
            bvec = cg_LS_lax(Xj,y-eta0)
            B0 = bvec.reshape(r,p1).T
            # --- B[1]  (p2,r) --- 
            Xj =  M.swapaxes(1,2) @ B0
            Xj = Xj.swapaxes(1,2).reshape(n,p2*r)  
            bvec = cg_LS_lax(Xj, y-eta0)
            B1 = bvec.reshape(r,p2).T
            # --- beta0  (p0,1) --- 
            eta = Xj @ bvec.reshape(-1,1) 
            beta0 = cg_LS_lax(X,y-eta).reshape(-1,1)
            # --- Converge? --- 
            yhat0 = y - eta - X @ beta0 
            devtmp = np.linalg.norm(yhat0)**2        
            diffdev = devtmp - dev0
            dev0 = devtmp
            abs_err = abs(diffdev)/abs(dev0)
            dev_list[rep, it] = dev0
            if verbose: 
                print("rep {:3d} iter {:3d} abs error {:10.7f} deviance {:.3f}".format(rep,it,abs_err,dev0))
            if (abs_err <TolFun) and (it > 5):
                nriters[rep]=it
                break

        # if it == MaxIter:
        #     print('Max iterations reached in replicate nr',rep)
        #     nriters[rep]=MaxIter
        if (dev0 < dev): # Record the smallest deviance
            etaest = eta
            beta0est = beta0
            B0est = B0
            B1est = B1
            dev = dev0
            yhat = yhat0
            best_rep = rep
        # if verbose: 
        #     print('replicate: ',best_rep)
        #     print(' iterates: ',it)
        #     print(' deviance: ',dev)
        #     print('    beta0: ',beta0est)

    return beta0est,B0est,B1est,dev_list,best_rep,nriters

def KTR(X,M,y,r,lams_set,solver_set,B0=[],B1=[],
        nonneg=1, MaxIter = 125, verbose = False):
    
    n,p0 = X.shape
    p = M.shape
    p1, p2, dev, TolFun = p[1], p[2], [], 5e-4
    if isinstance(B0, list):    # Initial Start (Random)
        B0 = 1-2*np.random.rand(p1,r)
        B1 = 1-2*np.random.rand(p2,r)
    
    beta0 = cg_LS_lax(X,y).reshape(-1,1)
    dev0 = np.linalg.norm(y - X @ beta0)**2

    for it in range(MaxIter):
        eta0 = X @ beta0
        # --- B[0]  (p1,r) --- 
        Xj = M @ B1
        Xj = Xj.swapaxes(1,2).reshape(n,p1*r)
        # bvec = cg_LS_lax(Xj,y-eta0)
        if solver_set[0] == 0: # 0. Choose Fused Lasso
            bvec = pga_FL_jax(Xj,y-eta0,B0.reshape(-1,1),lams_set[0,:],nonneg=nonneg,verbose=0)
        elif solver_set[0] == 1: # 0. Choose Fused Ridge
            bvec = pga_FR_jax(Xj,y-eta0,B0.reshape(-1,1),lams_set[0,:],nonneg=nonneg,verbose=0)
        if np.linalg.norm(bvec) < 1e-2: # Avoid all zeros
            bvec = cg_LS_lax(Xj, y-eta0)
        B0 = bvec.reshape(r,p1).T
        # --- B[1]  (p2,r) --- 
        Xj =  M.swapaxes(1,2) @ B0
        Xj = Xj.swapaxes(1,2).reshape(n,p2*r)  
        if solver_set[1] == 0: # 1. Choose Fused Lasso
            bvec = pga_FL_jax(Xj,y-eta0,B1.reshape(-1,1),lams_set[1,:],nonneg=nonneg,verbose=0)
        elif solver_set[1] == 1: # 1. Choose Fused Ridge
            bvec = pga_FR_jax(Xj,y-eta0,B1.reshape(-1,1),lams_set[1,:],nonneg=nonneg,verbose=0)
        if np.linalg.norm(bvec) < 1e-2: # Avoid all zeros
            bvec = cg_LS_lax(Xj, y-eta0)
        B1 = bvec.reshape(r,p2).T
        # --- beta0  (p0,1) --- 
        eta = Xj @ bvec.reshape(-1,1) 
        beta0 = cg_LS_lax(X,y-eta).reshape(-1,1)
        # --- Converge ? --- 
        yhat0 = y - eta - X @ beta0 
        devtmp = np.linalg.norm(yhat0)**2        
        diffdev = devtmp - dev0
        dev0 = devtmp
        abs_err = abs(diffdev)/abs(dev0)
        dev.append(dev0)
        # if verbose: 
            # print("rep {:3d} iter {:3d} abs error {:10.7f} deviance {:.3f}".format(rep,it,abs_err,dev0))
        if (abs_err <TolFun) and (it > 5):
            break

    return beta0,B0,B1,dev

'''
    ------------------------------ Slow Functions ------------------------------
'''
def SLOW_FUNCTIONS_DIVIDING_LINE():
    return True

def gd_LS(y, X, verbose = 0, delta = 1e-4):
    # Gradient Descent for Least Square
    # Written By WXJ, 0327-2023
    lsobj = lambda b: 0.5*np.linalg.norm(y-X@b)**2 
    max_iter = int(1e5) # No need to put this in the arguments
    S, c = X.T @ X, X.T @ y # Precomputing (1/2)
    a = 1.0 / np.max(np.real(np.linalg.eigvals(X.T @ X))) # Lipschitz Constant (2/2)
    b, b_old = np.random.randn(*c.shape), np.random.randn(*c.shape) # Initializaing b and b_old
    S = np.eye(X.shape[1]) - a * S

    def cond_fun(inputs):
        i, err, *_= inputs
        return (i < max_iter) & (err >= delta)
    def body_fun(inputs):
        i, err, b_old, b = inputs
        b = a * c + S @ b_old # Gradient Descent
        err = np.linalg.norm(b_old - b) / np.linalg.norm(b)
        b_old = b
        return (i+1, err, b_old, b)

    i, err = 0, 1.0
    inputs = (i, err, b_old, b) # Initializing
    while cond_fun(inputs):
        (i, err, b_old, b ) = body_fun(inputs) # Main function
        i += 1
        inputs = (i, err, b_old, b)

        if verbose == 1:
            objval = lsobj(b)
            print("iter {:2d}: error = {:.7f} obj. function = {:.7f}".format(i,err,objval))
    return b

def cg_LS(X, y, verbose = 0):
    '''
        Conjugate Gradient Method for Least Square 
        0.5 * ||y - Xb||^2 
        Inputs: 
            X: \in (m x n)
            y: \in (m)
        Outputs: 
            b0: \in (n)

        Written By WXJ, 0327-2023
    '''
    lsobj = lambda b: 0.5*np.linalg.norm(y-X@b)**2 

    max_iter, delta = int(1e5), 1e-4 
    key = jax.random.PRNGKey(0) # Setting random seed
    b0 = jax.random.uniform(key, shape=(X.T@y).shape)
    d0 = y - X @ b0
    r0 = X.T @ d0
    p0 = r0.copy()
    t0 = X @ p0

    def cond_fun(inputs):
        i, err, *_ = inputs
        return (i < max_iter) & (err >= delta)

    def body_fun(inputs):
        i, err, b0,d0,r0,p0,t0 = inputs
        alpha = (jnp.linalg.norm(r0)/jnp.linalg.norm(t0))**2
        b1 = b0 + alpha*p0    
        err = jnp.linalg.norm(b1-b0)/jnp.linalg.norm(b1) # Stopping Criteria
        d1 = d0 - alpha*t0
        r1 = X.T @ d1
        p1 = r1 + ((jnp.linalg.norm(r1)/jnp.linalg.norm(r0))**2)*p0
        t1 = X @ p1
        r0,t0,p0,d0,b0 = r1.copy(), t1.copy(),p1.copy(),d1.copy(),b1.copy()
        return (i+1, err, b0,d0,r0,p0,t0)

    i, err = 0, 1.0
    inputs = (i, err, b0,d0,r0,p0,t0)  # Initializing
    while cond_fun(inputs):
        (i, err, b0,d0,r0,p0,t0) = body_fun(inputs) # Main function
        inputs = (i, err, b0,d0,r0,p0,t0)

        if verbose == 1:
            objval = lsobj(b0)
            print("iter {:2d}: error = {:.7f} obj. function = {:.7f}".format(i,err,objval))

    return b0

def pga_FL_cond_body(y, X, b_init, lams, nonneg=1, verbose=0):
    '''
    Proximal Gradient Algorithm for Fused Lasso
    0.5 * || y - Xb ||^2 + lambda1*sum_(i=1)^p |b[i]| +lambda2*sum_(i=1)^(p-1) | b[i] - b[i+1] | 
    Inputs:
        y, X        (m), (m x n)
        b_init      (n)
    '''
    IfNonNeg = [-1e10, 0] # avoid boolean expression
    def flobj(y,X,b,lam1,lam2):
        funcvalue = 0.5*np.linalg.norm(y-X@b)**2+lam1*np.sum(np.abs(b))+lam2*np.sum(np.abs(b[1:]-b[:-1]))
        return funcvalue
    def soft(x, lam1):
        return np.sign(x)*np.maximum(np.abs(x)-lam1,0)
    max_iter,delta = int(1e5), 1e-4 # No need to put this in the arguments
    S, c = X.T @ X, X.T @ y # Precomputing (1/2)
    lam1, lam2 = lams[0], lams[1]
    a = 1.0 / np.max(np.real(np.linalg.eigvals(X.T @ X))) # Lipschitz Constant (2/2)
    b, b_old = np.random.randn(*c.shape), np.random.randn(*c.shape) # Initializaing b and b_old
    S = np.eye(X.shape[1]) - a * S
    def cond_fun(inputs):
        i, err, *_= inputs
        return (i < max_iter) & (err >= delta)
    def body_fun(inputs):
        i, err, b_old, b = inputs
        b =  np.clip(soft(ptv.tv1_1d(a*c+S@b_old,a*lam2),a*lam1),a_min=IfNonNeg[nonneg],a_max=None) # Gradient 
        err = np.linalg.norm(b_old - b) / np.linalg.norm(b)
        b_old = b
        return (i+1, err, b_old, b)
    i, err = 0, 1.0
    inputs = (i, err, b_old, b) # Initializing
    while cond_fun(inputs):
        (i, err, b_old, b ) = body_fun(inputs) # Main function
        i += 1
        inputs = (i, err, b_old, b)
        if verbose == 1:
            objval = flobj(y,X,b,lam1,lam2)
            print("iter {:2d}: error = {:.7f} obj. function = {:.7f}".format(i,err,objval))
    return b
