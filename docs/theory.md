We define the continuous Fourier transform (CFT) and its inverse as

$$
\begin{align}\label{eqn:FT}
    F(k) &= \hat{\mathcal{F}}_x[f(x)](k) = \left. \int_{-\infty}^\infty f(x)\exp(-i 2\pi \xi x) \d x \right|_{\xi = k}, \\
    f(x) &= \hat{\mathcal{F}}^{-1}_k[F(k)](x) = \left. \int_{-\infty}^\infty F(k)\exp(i 2\pi k \xi) \d k \right|_{\xi = x}.
\end{align}
$$

We use the usual definition of the fast Fourier transform (FFT)

$$
\begin{align}
    \v{A}[m] &= \Big(\text{fft}(\v{a})\Big)[m] =  \sum_{n=0}^{N-1} \v{a}[n] \exp\left( -i 2\pi \frac{mn}{N} \right), \\
    \v{a}[m] &= \Big(\text{ifft}(\v{A})\Big)[m] = \frac{1}{N} \sum_{n=0}^{N-1} \v{A}[n] \exp\left( +i2\pi \frac{mn}{N} \right).
\end{align}
$$