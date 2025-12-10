# NetKet Ising Mini-Project

Small exploration of the 1D transverse-field Ising model with NetKet. Includes mean-field ansätze, RBM VMC, and comparisons to exact diagonalization.

## Contents
- `modunetket.py`: Hilbert space and Ising Hamiltonian helpers (TFIM).
- `ising_mf.py`: Mean-field (real and complex) variational experiments.
- `ising_tests.py`: RBM VMC convergence vs exact ground state following the NetKet tutorial.
- `ising_ffn_tutorial.py`: Additional tutorial-based experiments (FFN/Jastrow style).

## Quickstart
1) (Optional) create an environment and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install netket jax matplotlib scipy tqdm
   ```
2) Run the RBM test:
   ```bash
   python ising_tests.py
   ```
   This will train an RBM, log energies, plot convergence, and print the exact ground-state energy.

## Notes
- Computations are set to CPU by default (`JAX_PLATFORM_NAME=cpu`) for portability.
- If matplotlib complains about missing `pyparsing`, install it with `pip install pyparsing`.
