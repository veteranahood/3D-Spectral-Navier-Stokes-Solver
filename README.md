# Gaia Protocol: GPU-Accelerated Spectral Navier–Stokes Solver

Gaia Protocol is a GPU-accelerated spectral solver for the 3D incompressible Navier–Stokes equations using a Leray projection to enforce incompressibility at machine precision.

The solver is designed for high-resolution turbulence experiments and emphasizes numerical stability, energy conservation, and spectral accuracy.

## Features
- Fully spectral DNS (FFT-based)
- Leray projection for incompressibility
- 2/3-rule de-aliasing
- GPU acceleration via CuPy
- Periodic cubic domain
- HDF5-ready data output (extensible)

## Mathematical Model
We solve the incompressible Navier–Stokes equations:

∂u/∂t + (u · ∇)u = −∇p + ν∇²u  
∇ · u = 0

Incompressibility is enforced via the Leray projection:

u ← P(u) = u − ∇Δ⁻¹(∇ · u)

## GPU Acceleration
All FFTs and array operations are executed on NVIDIA GPUs using CuPy.
A CUDA-capable GPU with CUDA 11.x or newer is required.

## Installation
```bash
pip install -r requirements.txt
