import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jaxtyping import Float, Int, Array, Complex, Num
from jax.numpy.fft import fft, ifft
Float2D = Float[Array, "x1 x2"]
Complex1D = Complex[Array, "x1"]
Array1D = Num[Array, "x1"]
Array2D = Num[Array, "x1 x2"]
pi = jnp.pi
exp = jnp.exp

def generate_bst(p: Array1D, q: Array1D) -> Complex1D:
    """
    Compute a discrete approximation of the 1D continuous Fourier transform
    using the Bluestein (chirp–z) algorithm. This involves embedding the linear convolution into a circular convolution by padding the arrays to a length twice that of the input.

    Important!
    For this routine to faithfully calculate a discrete approximation to a continuous Fourier transform, then the following assumptions are made:

        1. A are the samples of the original function at grid points:
        jnp.linspace(-Lp / 2, Lp / 2, len(a), endpoint=False)
        2. The ouput samples are similarly:
        jnp.linspace(-Lq / 2, Lq / 2, len(a), endpoint=False)
    """

    dp = jnp.mean(jnp.diff(p))
    dq = jnp.mean(jnp.diff(q))
    d = dp * dq
    p0 = p[0]
    q0 = q[0]
    N = len(p)
    n = jnp.arange(0, N)
    q_ramp = exp(-1j * 2 * pi * q0 * (p - p0))
    p_ramp = exp(-1j * 2 * pi * p0 * q)
    chirp = exp(-1j * pi * d * n**2)
    k = jnp.arange(2 * N)
    ak = jnp.where(k < N, k, k - 2 * N)
    kernel = exp(1j * pi * d * ak**2)

    @jax.jit
    def bst(a):
        a_chirped = jnp.pad(a * q_ramp * chirp, (0, N))
        conv = ifft(fft(a_chirped) * fft(kernel))[:N]
        return dp * p_ramp * chirp * conv

    return bst

def generate_bst2D(p1: Array1D, p2: Array1D, q1: Array1D, q2: Array1D):
    bst1 = generate_bst(p1, q1)
    bst2 = generate_bst(p2, q2)
    bst1_vm = jax.vmap(bst1, in_axes=1, out_axes=1)
    bst2_vm = jax.vmap(bst2, in_axes=0, out_axes=0)

    @jax.jit
    def bst2D(a2d: Array2D) -> Array2D:
        temp = bst1_vm(a2d)
        out  = bst2_vm(temp)
        return out

    return bst2D