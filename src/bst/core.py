import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jaxtyping import Float, Int, Array, Complex, Num
from jax.numpy.fft import fft, ifft
Float2D = Float[Array, "x1 x2"]
Complex1D = Complex[Array, "x1"]
Complex2D = Complex[Array, "x1 x2"]
Real1D = Float[Array, "x1"] | Int[Array, "x1"]
pi = jnp.pi
exp = jnp.exp
from typing import NamedTuple

class BSTPlan(NamedTuple):
    pre: Complex1D
    post: Complex1D
    kernel_fft: Complex1D
    N: int

def bst(t: Real1D, f: Real1D, a: Complex1D) -> Complex1D:
    r"""
    Approximate the continuous Fourier transform on an arbitrary uniform grid.

    This evaluates the quadrature approximation

    $$
    A(f_m) \approx \Delta t \sum_n a(t_n)
    \exp(-i 2\pi f_m t_n)
    $$

    using a Bailey--Swarztrauber / Bluestein chirp-convolution algorithm.

    Parameters
    ----------
    t
        Uniformly spaced input coordinates.
    f
        Uniformly spaced output frequency coordinates.
    a
        Samples of the input function on `t`.

    Returns
    -------
    Complex1D
        Approximate continuous Fourier transform sampled on `f`.

    See Also
    --------
    generate_bst : Precompute a reusable transform plan.
    ibst : Approximate inverse continuous Fourier transform.

    Notes
    -----
    The input and output coordinates must be uniformly spaced. Unlike the FFT-based
    approximation, the output spacing does not need to satisfy ``df = 1 / (N * dt)``.
    """
    plan = _generate_plan(t, f)
    return _execute_plan(plan, a)

def _generate_plan(t: Real1D, f: Real1D) -> BSTPlan:
    r"""
    test docstirn
    """
    N = len(t)
    if int(f.shape[0]) != int(N):
        raise ValueError("t and f must have the same length for this bst implementation.")
    
    dt = jnp.mean(jnp.diff(t))
    df = jnp.mean(jnp.diff(f))
    d = dt * df
    t0 = t[0]
    f0 = f[0]
    n = jnp.arange(0, N)
    f_ramp = exp(-1j * 2 * pi * f0 * (t - t0))
    t_ramp = exp(-1j * 2 * pi * t0 * f)
    chirp = exp(-1j * pi * d * n**2)
    k = jnp.arange(2 * N)
    ak = jnp.where(k < N, k, k - 2 * N)
    kernel = exp(1j * pi * d * ak**2)
    kernel_fft = fft(kernel)
    pre = f_ramp * chirp
    post = dt * t_ramp * chirp

    return BSTPlan(pre=pre, post=post, kernel_fft=kernel_fft, N=N)

def _execute_plan(plan: BSTPlan, a: Complex1D) -> Complex1D:
    N = plan.N
    a_chirped = jnp.pad(a * plan.pre, (0, N))
    conv = ifft(fft(a_chirped) * plan.kernel_fft)[:N]
    return plan.post * conv

def generate_bst(t: Real1D, f: Real1D):
    r"""
    test docstirn
    """
    plan = _generate_plan(t, f)
    def bst(a):
        return _execute_plan(plan, a)
    return bst

def ibst(t: Real1D, f: Real1D, A: Complex1D) -> Complex1D:
    plan = _generate_plan(f, t)
    dt = jnp.mean(jnp.diff(t))
    df = jnp.mean(jnp.diff(f))
    return jnp.conj(_execute_plan(plan, jnp.conj(A))) * dt / df

def generate_ibst(t: Real1D, f: Real1D):
    plan = _generate_plan(f, t)
    def ibst(A):
        return jnp.conj(_execute_plan(plan, jnp.conj(A)))
    return ibst

def bst2D(t1: Real1D, t2: Real1D, f1: Real1D, f2: Real1D, a: Complex2D):
    op = generate_bst2D(t1, t2, f1, f2)
    return op(a)

def generate_bst2D(t1: Real1D, t2: Real1D, f1: Real1D, f2: Real1D):
    bst1 = generate_bst(t1, f1)
    bst2 = generate_bst(t2, f2)
    bst1_vm = jax.vmap(bst1, in_axes=1, out_axes=1)
    bst2_vm = jax.vmap(bst2, in_axes=0, out_axes=0)

    def bst2D(a: Complex2D):
        return bst2_vm(bst1_vm(a))

    return bst2D

def generate_ibst2D(t1: Real1D, t2: Real1D, f1: Real1D, f2: Real1D):
    bst2 = generate_bst2D(f1, f2, t1, t2)

    def ibst2D(A: Complex2D):
        return jnp.conj(bst2(jnp.conj(A)))

    return ibst2D