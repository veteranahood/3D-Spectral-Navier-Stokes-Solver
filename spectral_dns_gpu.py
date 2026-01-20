"""
Gaia Protocol: 3D Spectral Navier-Stokes Solver (GPU)
Author: Anthony Scott Hood
License: MIT
Description: High-performance Pseudo-Spectral solver for the 3D Incompressible 
Navier-Stokes equations using CuPy (GPU acceleration). Implements Leray Projection 
and Orszag 2/3 de-aliasing for turbulence simulation.
"""

import cupy as cp        # GPU arrays
import numpy as np       # CPU arrays for saving
import matplotlib.pyplot as plt
import time
import h5py

# --- CONFIGURATION ---
N = 128          # Grid Resolution (128^3)
L = 2 * cp.pi    # Domain Size
dt = 0.001       # Time Step
steps = 1000     # Simulation Duration
nu = 0.002       # Viscosity (Low = Turbulent)

print(f"Initializing Gaia 3D Solver: {N}x{N}x{N} | Viscosity: {nu}")

# --- SPECTRAL GRID SETUP ---
k = 2 * cp.pi * cp.fft.fftfreq(N, L/N)
KX, KY, KZ = cp.meshgrid(k, k, k, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0,0,0] = 1.0  # Avoid division by zero

# Orszag 2/3 De-aliasing Rule
k_max = N / 2
k_cut = int(cp.floor(k_max * 2/3))
dealias_mask = (cp.abs(KX) < k_cut) & (cp.abs(KY) < k_cut) & (cp.abs(KZ) < k_cut)

# --- CORE MATH FUNCTIONS ---

def leray_projection(u_hat):
    """
    Project velocity field onto divergence-free manifold (The Dimensional Accord).
    Removes the compressive component (Sound waves) to enforce incompressibility.
    """
    div_hat = 1j*KX*u_hat[...,0] + 1j*KY*u_hat[...,1] + 1j*KZ*u_hat[...,2]
    p_hat = div_hat / K2
    
    u_hat[...,0] -= 1j*KX*p_hat
    u_hat[...,1] -= 1j*KY*p_hat
    u_hat[...,2] -= 1j*KZ*p_hat
    
    return u_hat

def compute_rhs(u_hat):
    """
    Calculates the Right-Hand Side of Navier-Stokes:
    RHS = - (u . grad) u + nu * laplacian(u)
    """
    # 1. De-alias and Transform to Real Space
    u_hat_de = u_hat * dealias_mask[..., None]
    u_real = cp.real(cp.fft.ifftn(u_hat_de, axes=(0,1,2)))
    
    # 2. Compute Gradients in Spectral Space
    grad_x_hat = 1j * KX[..., None] * u_hat_de
    grad_y_hat = 1j * KY[..., None] * u_hat_de
    grad_z_hat = 1j * KZ[..., None] * u_hat_de
    
    # 3. Transform Gradients to Real Space
    grad_x = cp.real(cp.fft.ifftn(grad_x_hat, axes=(0,1,2)))
    grad_y = cp.real(cp.fft.ifftn(grad_y_hat, axes=(0,1,2)))
    grad_z = cp.real(cp.fft.ifftn(grad_z_hat, axes=(0,1,2)))
    
    # 4. Compute Non-Linear Advection in Real Space: (u . grad) u
    adv_real = (u_real[...,0,None] * grad_x + 
                u_real[...,1,None] * grad_y + 
                u_real[...,2,None] * grad_z)
    
    # 5. Transform Advection back to Spectral & De-alias
    adv_hat = cp.fft.fftn(adv_real, axes=(0,1,2)) * dealias_mask[..., None]
    
    # 6. Diffusion Term (Exact in Spectral Space)
    diff_hat = -nu * K2[..., None] * u_hat
    
    # 7. Combine and Project
    rhs_hat = diff_hat - adv_hat
    return leray_projection(rhs_hat)

def compute_enstrophy(u_hat):
    """Calculates total Enstrophy (Z) to verify smoothness."""
    omega_x = 1j * (KY*u_hat[...,2] - KZ*u_hat[...,1])
    omega_y = 1j * (KZ*u_hat[...,0] - KX*u_hat[...,2])
    omega_z = 1j * (KX*u_hat[...,1] - KY*u_hat[...,0])
    return 0.5 * cp.mean(cp.abs(omega_x)**2 + cp.abs(omega_y)**2 + cp.abs(omega_z)**2)

# --- INITIALIZATION (Chaotic Field) ---
print("Generating chaotic initial conditions...")
# Random phase initialization with specific energy spectrum
u_hat = cp.random.randn(N, N, N, 3) + 1j * cp.random.randn(N, N, N, 3)
# Filter high frequencies for smooth start
u_hat *= (K2 < (N/4)**2)[..., None]
u_hat = leray_projection(u_hat)

# --- MAIN LOOP (RK4 Integration) ---
print(f"Starting GPU Simulation ({steps} steps)...")
start_time = time.time()
energy_history = []
enstrophy_history = []

for t in range(steps):
    k1 = compute_rhs(u_hat)
    k2 = compute_rhs(u_hat + 0.5 * dt * k1)
    k3 = compute_rhs(u_hat + 0.5 * dt * k2)
    k4 = compute_rhs(u_hat + dt * k3)
    
    u_hat = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Diagnostics every 10 steps
    if t % 10 == 0:
        # Calculate Energy and Enstrophy
        u_real = cp.real(cp.fft.ifftn(u_hat, axes=(0,1,2)))
        E = 0.5 * cp.mean(cp.sum(u_real**2, axis=-1))
        Z = compute_enstrophy(u_hat)
        
        energy_history.append(cp.asnumpy(E))
        enstrophy_history.append(cp.asnumpy(Z))
        print(f"Step {t}: Energy={E:.6f} | Enstrophy={Z:.6f}")
        
        # Stability Check
        if cp.isnan(E) or E > 1e5:
            print("Instability detected! Stopping.")
            break

total_time = time.time() - start_time
print(f"Simulation Complete. Time: {total_time:.2f}s")

# --- SAVE DATA ---
print("Saving 3D Turbulence Data to HDF5...")
u_final = cp.asnumpy(cp.real(cp.fft.ifftn(u_hat, axes=(0,1,2))))
with h5py.File('gaia_turbulence_3d.h5', 'w') as f:
    f.create_dataset('velocity', data=u_final)
    f.attrs['Author'] = 'Anthony Scott Hood'
    f.attrs['Description'] = 'Gaia Protocol 3D Turbulence'

print("Done.")
