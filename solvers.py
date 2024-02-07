import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def symmetric_permutation(A,p):
    A = A.tocsc()
    invp=[0 for _ in p]
    for i,pi in enumerate(p):
        invp[pi]=i
    rids=[]
    cids=[]
    vals=[]
    for c in p:
        beg=A.indptr[c]
        end=A.indptr[c+1]
        for i in range(beg,end):
            r=A.indices[i]
            v=A.data[i]
            rids.append(invp[r])
            cids.append(invp[c])
            vals.append(v)
    return sp.coo_array((vals,(rids,cids)),A.shape)




def random_uniform_symm_posdef(m,nnz,std,rng=None):    
    if rng is None:    
        rng=np.random.default_rng(0)    
    rids=[]    
    cids=[]    
    vals=[]
    #Create a random sparse upper triangular matrix
    #with nonzeros on each row i is pulled from a normal
    #distribution from mean i. Nonzeros less than i are mirrored
    #across i to maintain upper triangular property.
    for i in range(m):
        cs=[i+abs(int(rng.normal(loc=0,scale=std))) for _ in range(nnz)]
        cs=[min(max(0,ci),m-1) for ci in cs]
        cs=set(cs)    
        for ci in cs:    
            rids.append(i)    
            cids.append(ci)    
            vals.append(rng.uniform(-1,1))    
    U=sp.coo_array((vals,(rids,cids)),shape=(m,m))
    A = U + U.T
    #A is symmetric but indefinite because of 0s on the diagonal,
    #so we shift the diagonal to guarantee all positive eigenvalues.
    #Choose a random scaling so that we get systems of varying
    #conditioning
    d = abs(A)@np.ones(m) + rng.choice([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])*rng.uniform(0.9,1.1)
    return A + sp.diags([d],[0])



def cg_noprecon(A,x):
    m,_=A.shape
    read_writes=0
    def evalA(x):
        nonlocal read_writes
        read_writes+=A.nnz
        read_writes+=2*x.size
        return A@x

    errs=[]
    ops=[]
    def callback(xk):
        nonlocal errs
        nonlocal ops
        errs.append(np.linalg.norm((xk-x)/x,ord=np.inf))
        ops.append(read_writes)
    b = A@x
    spla.cg(spla.LinearOperator((m,m),matvec=evalA),b,callback=callback,tol=1e-14)
    return ops,errs
