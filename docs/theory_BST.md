See the page [Conventions](conventions.md) for details on the notations used, the Fourier conventions, and other definitions used here. See also the page [Approximation with DFT](theory_DFT.md) which covers how to approximate the continuous Fourier transform with the discrete Fourier transform. The derivations there are also very relevant to this section, and could help to make things clearer.

## The objective

Given a function $f(x)$ that is negligible outside the interval $(-L_x/2, L_x/2)$, we want to compute a numerical approximation to its exact continuous Fourier transform (CFT). Importantly, we want to be able to choose the output grid $\v{k}$ in reciprocal space independently from the input grid $\v{x}$ in direct space. As we saw in the previous [theory page](theory_DFT.md) this is not possible with the DFT which intrinsically links the input and output grids. In addition 
