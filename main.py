import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solvers


seed=2387
rng = np.random.default_rng(seed)
m=1024
std=128
nnz=3


A = solvers.random_uniform_symm_posdef(m,nnz,std,rng=rng)
x = rng.uniform(-1,1,size=m)

#plt.spy(A,markersize=1)
#plt.savefig("spyA.svg")

ops,errs = solvers.cg_noprecon(A,x)

plt.semilogy([op/(A.nnz) for op in ops],errs,label = "Unpreconditioned CG")
plt.title("Cost-error curve")
plt.xlabel("Effective evaluations of A (total mem ops)/(nonzeros(A))")
plt.ylabel("Maximum relative error: max(|x-xh|/|x|)")
plt.legend()
plt.savefig("errs.svg")
