# %%
# ising_ffn_tutorial.py
#
# Ground-state of 1D transverse-field Ising model with NetKet,
# following the "gs-ising" netket tutorial using an FFN ansatz.
#  Besides, we compute the S2 with netket and homemade function. Also the gradient

# THE CODE IS SELFCONTAINED. SO FAR NO MODULE IS BUILT FOR THE MAIN.  IT IS  AN EXERCISE AND POLISHING SEEMS MANDATORY.

# David, December 2025
#
import os
import matplotlib.pyplot as plt
import netket as nk
import jax.numpy as jnp
import flax.linen as nn
from netket.operator.spin import sigmax, sigmaz

# ------
#  Backend: stay on CPU (optional, but good for reproducibility)
# ------
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# ------
#  Hamiltonian: 1D Ising chain with periodic boundary conditions
# ------
N = 16  # number of spins
Gamma = -1.0  # transverse field
V = -1.0  # nearest-neighbour coupling

hi = nk.hilbert.Spin(s=1 / 2, N=N)

# ising Hamiltonian
H = 0
for i in range(N):
    H += Gamma * sigmax(hi, i)
    H += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

# ED GS energy (NetKet Lanczos)
E_gs = float(nk.exact.lanczos_ed(H, k=1)[0])
print(f"Exact ground-state energy (Lanczos): {E_gs:.12f}")


# --------
# FFN ansatz as in the tutorial
# ---------
class FFN(nn.Module):
    """
    Simple feed-forward network for log ψ(σ).
    """

    @nn.compact
    def __call__(self, x):
        # x has shape (batch, N) with entries ±1
        x = nn.Dense(features=16)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=16)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        # Return scalar per sample: log ψ(σ)
        x = jnp.squeeze(x, axis=-1)
        return x


model = FFN()

# -------
# Variational state: Monte Carlo state with MetropolisLocal sampler
# -------
sampler = nk.sampler.MetropolisLocal(hi)

vstate = nk.vqs.MCState(
    sampler,
    model,
    n_samples=1008,
)

# ------
# Optimizer + SR + VMC driver
# ------

# in the tutorial learning_rate=0.05
optimizer = nk.optimizer.Sgd(learning_rate=0.1)

sr = nk.optimizer.SR(diag_shift=0.1)

# following the tutorial, i build the optimisation driver.
vmc = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=sr,
)

log = nk.logging.RuntimeLog()

vmc.run(n_iter=300, out=log)

data = log.data

# to know wich is stored in log.data
# print(log.data.keys())
# in our case dict_keys(['acceptance', 'Energy'])

# ------
# Plot energy vs iterations + exact GS line
# ------
plt.figure(figsize=(8, 4))
plt.errorbar(
    data["Energy"].iters,
    data["Energy"].Mean,
    yerr=data["Energy"].Sigma,
    label="FFN VMC",
)
plt.hlines(
    E_gs,
    xmin=0,
    xmax=data["Energy"].iters[-1],
    colors="black",
    linestyles="dashed",
    label="Exact",
)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("1D TFIM FFN vs exact ground state")
plt.legend()
plt.tight_layout()
plt.show()

# -------
#  Print final energies compare ED and FFN
# -------
print(f"Exact GS energy  : {E_gs:.12f}")
print(
    f"FFN final energy : {data['Energy'].Mean[-1]:.12f} ± {data['Energy'].Sigma[-1]:.3e}"
)


# %%

###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################

# --------
#  S2 and its gradient
# ---------
import jax
from jax.flatten_util import ravel_pytree

NA = N // 2  # size of subsystem A


def renyi2_from_samples(params, sigma1, sigma2):
    """
    Monte Carlo estimator of S2(A) for half-chain A using two
    *independent* replica batches sigma1, sigma2 ~ Π.

    sigma1, sigma2: arrays of shape (M., N).
    the formula for the S2 is given by
    S2 = - log2 < \psi(sigma, eta' ) \psi(sigma', eta) / < \psi(sigma, eta) \psi(sigma', eta') >
    """
    # Make same length and even
    M = min(sigma1.shape[0], sigma2.shape[0])
    if M % 2 == 1:
        M -= 1
    # sigma1 and sigma 2 will be given by samples, i.e.
    # samples1 = vstate.sample(n_samples=n_samples)
    # samples2 = vstate.sample(n_samples=n_samples)
    # notice that sigma dimension is(M1, N), so sigma1[:M] is (M,N)
    sigma1 = sigma1[:M]
    sigma2 = sigma2[:M]

    # Split each replica into A and B
    alpha = sigma1[:, :NA]  # first NA columns (columns 0..NA-1)
    beta = sigma1[:, NA:]  # remaining columns (columns NA..N-1)
    alpha_p = sigma2[:, :NA]
    beta_p = sigma2[:, NA:]

    # Swapped configurations (α',β) and (α,β')
    sigma_a_p_b = jnp.concatenate([alpha_p, beta], axis=-1)
    sigma_a_b_p = jnp.concatenate([alpha, beta_p], axis=-1)

    def logpsi_batch(p, σ):
        # log ψ(σ) from the same FFN model as vstate
        return model.apply({"params": p}, σ)

    # Log-amplitudes for original and swapped configs
    logpsi_ab = logpsi_batch(params, sigma1)
    logpsi_a_p_b_p = logpsi_batch(params, sigma2)
    logpsi_a_p_b = logpsi_batch(params, sigma_a_p_b)
    logpsi_a_b_p = logpsi_batch(params, sigma_a_b_p)

    # Hastings swap estimator:
    # Q_loc = ψ(α',β) ψ(α,β') / [ψ(α,β) ψ(α',β')]
    log_q_loc = (logpsi_a_p_b + logpsi_a_b_p) - (logpsi_ab + logpsi_a_p_b_p)
    q_loc = jnp.exp(log_q_loc)

    swap_est = jnp.mean(q_loc)
    # same convention as NetKet: log base 2
    # notice that jnp.log es the natural logarithm. thus, we dvide by ln(2)
    S2 = -jnp.log(swap_est) / jnp.log(2.0)
    return S2.real


# ------
#  Helper function to compute S2 (MC + native)
#  and gradient with same replicas
# ------
from netket.experimental.observable import Renyi2EntanglementEntropy


def compute_S2_all(vstate, n_samples=100000):
    """
    Compute:
      - S2 (homemade MC swap estimator),
      - its gradient w.r.t. parameters,
      - S2 from NetKet's native Renyi2EntanglementEntropy,

    using two *independent* replica batches drawn from vstate.
    """
    # Draw two independent replica batches
    samples1 = vstate.sample(n_samples=n_samples)
    samples2 = vstate.sample(n_samples=n_samples)

    samples1 = samples1.reshape(-1, N)
    samples2 = samples2.reshape(-1, N)

    # Homemade S2 and its gradient (fixed replicas)
    ###############################################################
    # Explanation of the syntax:
    #
    #     jax.value_and_grad(renyi2_from_samples)(
    #         vstate.parameters, samples1, samples2
    #     )
    #
    # 1. JAX’s value_and_grad expects a *function*, not the output
    #    of a function. Therefore we must pass the function object:
    #
    #       jax.value_and_grad(renyi2_from_samples)
    #
    #    and NOT something like renyi2_from_samples(...) which is
    #    already a number. JAX cannot differentiate numbers.
    #
    # 2. value_and_grad(f) returns a NEW function g such that:
    #
    #         g(theta, ...) = ( f(theta,...),  ∂f/∂theta )
    #
    #    That is: the returned function computes both the value of f
    #    AND the gradient with respect to the *first* argument.
    #
    # 3. So the full call:
    #
    #     jax.value_and_grad(renyi2_from_samples)(
    #         vstate.parameters, samples1, samples2
    #     )
    #
    #    is equivalent to:
    #
    #       g = jax.value_and_grad(renyi2_from_samples)
    #       S2_val, grad_S2 = g(vstate.parameters, samples1, samples2)
    #
    ###############################################################
    S2_val, grad_S2 = jax.value_and_grad(renyi2_from_samples)(
        vstate.parameters, samples1, samples2
    )

    # The gradient grad_S2 is a PyTree containing all parameter gradients
    # (one array per layer/kernel/bias). To inspect it, we first flatten
    # the whole PyTree into a single 1-D vector using ravel_pytree.
    #
    #   grad_dense: all gradients concatenated into one vector
    #   _         : function to rebuild the PyTree (unused here)
    #
    # Then we compute the L2 norm of this flattened gradient vector:
    #
    #   ||∂S2/∂θ||_2 = sqrt( sum_i (grad_i)^2 ).
    #
    # This gives a single scalar measuring the overall gradient magnitude.
    grad_dense, _ = ravel_pytree(grad_S2)
    grad_norm = jnp.linalg.norm(grad_dense)

    # Native NetKet S2 (value only)
    S2_native_op = Renyi2EntanglementEntropy(hi, list(range(NA)))
    S2_native_stats = vstate.expect(S2_native_op)
    S2_native = float(S2_native_stats.mean)

    print(f"[compute_S2_all] S2 (MC estimator)      : {float(S2_val):.6f}")
    print(f"[compute_S2_all] ||∂S2/∂θ|| (L2 norm)   : {float(grad_norm):.6e}")
    print(f"[compute_S2_all] S2 (NetKet native)     : {S2_native:.6f}")

    return S2_val, grad_S2, grad_norm, S2_native


# Example call after training:
S2_val, grad_S2, grad_norm, S2_native = compute_S2_all(vstate, n_samples=50000)


# %%
