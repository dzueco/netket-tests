# %%
import os
import netket as nk
import sys
import matplotlib.pyplot as plt
from netket.optimizer import Sgd

import jax

from tqdm import tqdm
from scipy.sparse.linalg import eigsh

""" esto es para añadir el actual directorio y poder impotar módulos """
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import reload

import modunetket as mdnkt

mdnkt = reload(mdnkt)


# %%
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# size, hilbert space and hamiltonian:
N = 12
hi = nk.hilbert.Spin(s=1 / 2, N=N, inverted_ordering=False)
# NoTE: Why I write "inverted_ordering=False)" (see documentation)

H = mdnkt.h_ising1d(N, hi, 2, -1)

sp_h = H.to_sparse()
sp_h.shape

eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
print("eigenvalues with scipy sparse:", eig_vals)
E_gs = eig_vals[0]

# %%

# two models: mf and mf_complex
sampler = nk.sampler.MetropolisLocal(hi)

mf_model = mdnkt.MF()
vstate = nk.vqs.MCState(sampler, mf_model, n_samples=512)

mfc_model = mdnkt.MF_complex()
vstatec = nk.vqs.MCState(sampler, mfc_model, n_samples=512)

energy_history_mf = []
lambda_history_mf = []

energy_history_mfc = []
lambda_history_mfc = []
phi_history_mfc = []

# Número de pasos
n_steps = 400

# Optimización para ambos modelos
for i in tqdm(range(n_steps)):
    # MF Model (sin parte imaginaria)
    E, E_grad = vstate.expect_and_grad(H)
    energy_history_mf.append(E.mean.real)
    lambda_current = vstate.parameters["lambda"]
    lambda_history_mf.append(lambda_current)
    new_pars = jax.tree_util.tree_map(
        lambda x, y: x - 0.05 * y, vstate.parameters, E_grad
    )
    vstate.parameters = new_pars

    # MFC Model (con parte imaginaria)
    Ec, E_gradc = vstatec.expect_and_grad(H)
    energy_history_mfc.append(Ec.mean.real)
    lambda_current_c = vstatec.parameters["lambda"]
    phi_current_c = vstatec.parameters["phi"]
    lambda_history_mfc.append(lambda_current_c)
    phi_history_mfc.append(phi_current_c)
    new_pars_c = jax.tree_util.tree_map(
        lambda x, y: x - 0.05 * y, vstatec.parameters, E_gradc
    )
    vstatec.parameters = new_pars_c

# Graficar resultados
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Energía
axs[0].set_title("Energy Evolution")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Energy")
axs[0].plot(energy_history_mf, label="MF (Real Only)")
axs[0].plot(energy_history_mfc, label="MFC (Complex)")
axs[0].axhline(E_gs, color="black", linestyle="--", label="Exact energy")
axs[0].legend()

# Lambda
axs[1].set_title("Lambda Evolution")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Lambda")
axs[1].plot(lambda_history_mf, label="MF (Real Only)", color="orange")
axs[1].plot(lambda_history_mfc, label="MFC (Complex)", color="blue")
axs[1].legend()

# Phi (solo para MFC)
axs[2].set_title("Phi Evolution")
axs[2].set_xlabel("Iterations")
axs[2].set_ylabel("Phi (mod 2π)")
axs[2].plot(phi_history_mfc, label="MFC (Complex)", color="green")
axs[2].legend()

plt.tight_layout()
plt.show()


# %%


vstate.init_parameters()
vstatec.init_parameters()


optimizer = nk.optimizer.Sgd(learning_rate=0.05)

gs_mf = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)
gs_mfc = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstatec,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)


log_mf = nk.logging.RuntimeLog()
log_mfc = nk.logging.RuntimeLog()


n_steps = 300
gs_mf.run(n_iter=n_steps, out=log_mf)
gs_mfc.run(n_iter=n_steps, out=log_mfc)


data_mf = log_mf.data
data_mfc = log_mfc.data


plt.figure(figsize=(15, 5))


plt.errorbar(
    data_mf["Energy"].iters,
    data_mf["Energy"].Mean,
    yerr=data_mf["Energy"].Sigma,
    label="MF SGD ",
)
plt.plot(
    range(len(energy_history_mf)),
    energy_history_mf,
    label="MF Previous Method",
    linestyle="dashed",
)

# Energía para MFC
plt.errorbar(
    data_mfc["Energy"].iters,
    data_mfc["Energy"].Mean,
    yerr=data_mfc["Energy"].Sigma,
    label="MFC SGD ",
)
plt.plot(
    range(len(energy_history_mfc)),
    energy_history_mfc,
    label="MFC Previous Method",
    linestyle="dashed",
)


plt.axhline(E_gs, color="black", linestyle="--", label="Exact Energy")


plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("Energy Comparison: Models and Methods")
plt.legend()
plt.show()
# %%
