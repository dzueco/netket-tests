import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from netket.operator.spin import sigmax, sigmaz


def h_ising1d(N, hi, Gamma, V):
    """provides the Hamiltonian for the ising in a tranvserse field (1D, nn)

    Args:
        N (integer): Size (number of spins)
        hi (netket): hilbert space
        Gamma (float): magnetic field
        V (_type_): interaction strenght

    Returns:
        _type_: Hamiltonian
    """

    H = sum([sigmax(hi, i) for i in range(N)])

    H = Gamma * H

    H += sum([V * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])

    return H


# A Flax model must be a class subclassing `nn.Module`
class MF(nn.Module):

    # The most compact way to define the model is this.
    # The __call__(self, x) function should take as
    # input a batch of states x.shape = (n_samples, L)
    # and should return a vector of n_samples log-amplitudes
    @nn.compact
    def __call__(self, x):

        # A tensor of variational parameters is defined by calling
        # the method `self.param` where the arguments will be:
        # - arbitrary name used to refer to this set of parameters
        # - an initializer used to provide the initial values.
        # - The shape of the tensor
        # - The dtype of the tensor.
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)

        # compute the probabilities
        p = nn.log_sigmoid(lam * x)

        # sum the output
        return 0.5 * jnp.sum(p, axis=-1)


class MF_complex(nn.Module):
    """
    Mean-Field Ansatz with complex coefficients.
    """

    @nn.compact
    def __call__(self, x):
        # Real part (same as the original implementation)
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)
        p_real = nn.log_sigmoid(lam * x)
        real_part = 0.5 * jnp.sum(p_real, axis=-1)

        # Imaginary part (phase contribution)
        phi = self.param("phi", nn.initializers.normal(), (1,), float)
        # Restricción de phi al rango [0, 2π)
        phase = jnp.pi * nn.sigmoid(phi)

        # Fase aplicada solo cuando x = 1
        imag_part = jnp.sum(phase * (x == 1), axis=-1)

        # Combine real and imaginary parts to form complex log-amplitudes
        return real_part + 1j * imag_part


class model_tanh(nn.Module):

    # The most compact way to define the model is this.
    # The __call__(self, x) function should take as
    # input a batch of states x.shape = (n_samples, L)
    # and should return a vector of n_samples log-amplitudes
    @nn.compact
    def __call__(self, x):

        # A tensor of variational parameters is defined by calling
        # the method `self.param` where the arguments will be:
        # - arbitrary name used to refer to this set of parameters
        # - an initializer used to provide the initial values.
        # - The shape of the tensor
        # - The dtype of the tensor.
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)

        # compute the probabilities
        p = (1 + nn.tanh(lam * x)) / 2
        log_p = jnp.log(p)
        return 0.5 * jnp.sum(log_p, axis=-1)
