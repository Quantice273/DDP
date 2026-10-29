import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for saved PNGs
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)


# ==============================================================================
# EXPERIMENT 1: Particle Near Wall Under Gravity (D(z) = gamma * z)
# ==============================================================================
def sim_exp1(N_particles=20000, dt=0.001, N_steps=3000):
    z_with = np.ones(N_particles) * 2.0  # initial z0 = 2.0
    z_no   = np.ones(N_particles) * 2.0
    
    t_hist, m1_with, m2_with = [], [], []
    m1_no, m2_no = [], []
    sq_dt = np.sqrt(dt)
    t = 0.0
    
    for step in range(N_steps):
        if step % 10 == 0:
            t_hist.append(t)
            m1_with.append(np.mean(z_with))
            m2_with.append(np.mean(z_with**2))
            m1_no.append(np.mean(z_no))
            m2_no.append(np.mean(z_no**2))
            
        dW = np.random.normal(0, sq_dt, N_particles)
        
        # With f1 drift (+1)
        z_curr = np.maximum(z_with, 1e-6)
        dz_with = (1.0 - z_curr) * dt + np.sqrt(2.0 * z_curr) * dW + 0.5 * (dW**2 - dt)
        z_with = np.maximum(z_curr + dz_with, 1e-6)
        
        # Without f1 drift (0)
        z_curr_no = np.maximum(z_no, 1e-6)
        dz_no = (-z_curr_no) * dt + np.sqrt(2.0 * z_curr_no) * dW + 0.5 * (dW**2 - dt)
        z_no = np.maximum(z_curr_no + dz_no, 1e-6)
        
        t += dt
        
    return z_with, z_no, np.array(t_hist), np.array(m1_with), np.array(m2_with), np.array(m1_no), np.array(m2_no)

z_with, z_no, t1, m1_with_1, m2_with_1, m1_no_1, m2_no_1 = sim_exp1()

# Plot Exp 1 Density Histogram
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, 5, 60)
z_analytical = np.linspace(0.01, 5, 200)
p_analytical = np.exp(-z_analytical)

sns.histplot(z_with, bins=bins, stat='density', label=r'Simulation (With Spurious Drift $f_1=+1$)', color='#2b5c8f', alpha=0.6, ax=ax)
sns.histplot(z_no, bins=bins, stat='density', label=r'Simulation (Without Drift $f_1=0$)', color='#d95f02', alpha=0.4, ax=ax)
ax.plot(z_analytical, p_analytical, 'k--', linewidth=2.5, label=r'Analytical Boltzmann Target $P_{eq}(z) = e^{-z}$')

ax.set_title(r'Experiment 1: Particle Near Wall under Gravity ($D(z) = \gamma z$)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(r'Dimensionless Distance from Wall $\tilde{z}$')
ax.set_ylabel(r'Probability Density $P(\tilde{z})$')
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.2)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(output_dir, 'exp1_wall_gravity.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot Exp 1 Moments
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
z0_1 = 2.0
m1_target_1 = z0_1 * np.exp(-t1) + (1.0 - np.exp(-t1))
m2_target_1 = z0_1**2 * np.exp(-2*t1) + 4*(z0_1 - 1)*(np.exp(-t1) - np.exp(-2*t1)) + 2*(1 - np.exp(-2*t1))

ax1.plot(t1, m1_with_1, 'o', color='#2b5c8f', markersize=3, markevery=15, label=r'Simulation (With Drift $f_1=+1$)')
ax1.plot(t1, m1_target_1, 'k--', linewidth=2, label=r'Analytical Target: $1 + (z_0-1)e^{-t}$')
ax1.plot(t1, m1_no_1, 's', color='#d95f02', markersize=3, markevery=15, label=r'Simulation (Without Drift $f_1=0$)')
ax1.set_title(r'First Moment $\langle \tilde{z}(\tilde{t}) \rangle$', fontsize=13, fontweight='bold')
ax1.set_xlabel(r'Dimensionless Time $\tilde{t}$')
ax1.set_ylabel(r'$\langle \tilde{z} \rangle$')
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

ax2.plot(t1, m2_with_1, 'o', color='#2b5c8f', markersize=3, markevery=15, label=r'Simulation (With Drift $f_1=+1$)')
ax2.plot(t1, m2_target_1, 'k--', linewidth=2, label=r'Analytical Equilibrium Target: $2.0$')
ax2.plot(t1, m2_no_1, 's', color='#d95f02', markersize=3, markevery=15, label=r'Simulation (Without Drift $f_1=0$)')
ax2.set_title(r'Second Moment $\langle \tilde{z}^2(\tilde{t}) \rangle$', fontsize=13, fontweight='bold')
ax2.set_xlabel(r'Dimensionless Time $\tilde{t}$')
ax2.set_ylabel(r'$\langle \tilde{z}^2 \rangle$')
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

fig.suptitle(r'Experiment 1: Moment Relaxation Near Wall ($D(z) = \gamma z$)', fontsize=15, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(output_dir, 'exp1_z_moments.png'), dpi=150, bbox_inches='tight')
plt.close()


# ==============================================================================
# EXPERIMENT 2: Particle Bounded Between Parallel Walls (D(x) = 1 - x^2)
# ==============================================================================
def sim_exp2(N_particles=20000, dt=0.0005, N_steps=3000):
    x_0 = 0.6
    x_with = np.ones(N_particles) * x_0
    x_no   = np.ones(N_particles) * x_0
    
    t_hist, m1_with, m2_with = [], [], []
    m1_no, m2_no = [], []
    sq_dt = np.sqrt(dt)
    t = 0.0
    
    for step in range(N_steps):
        if step % 10 == 0:
            t_hist.append(t)
            m1_with.append(np.mean(x_with))
            m2_with.append(np.mean(x_with**2))
            m1_no.append(np.mean(x_no))
            m2_no.append(np.mean(x_no**2))
            
        dW = np.random.normal(0, sq_dt, N_particles)
        
        # With f1 drift (-2x)
        x_c = np.clip(x_with, -0.999, 0.999)
        g = np.sqrt(2.0 * (1.0 - x_c**2))
        g_prime = -2.0 * x_c / (g + 1e-9)
        dx_with = (-2.0 * x_c) * dt + g * dW + 0.5 * g * g_prime * (dW**2 - dt)
        x_with = np.clip(x_c + dx_with, -0.999, 0.999)
        
        # Without f1 drift
        x_c_no = np.clip(x_no, -0.999, 0.999)
        g_no = np.sqrt(2.0 * (1.0 - x_c_no**2))
        g_prime_no = -2.0 * x_c_no / (g_no + 1e-9)
        dx_no = 0.0 * dt + g_no * dW + 0.5 * g_no * g_prime_no * (dW**2 - dt)
        x_no = np.clip(x_c_no + dx_no, -0.999, 0.999)
        
        t += dt
        
    return x_with, x_no, np.array(t_hist), np.array(m1_with), np.array(m2_with), np.array(m1_no), np.array(m2_no), x_0

x_with, x_no, t2, m1_with_2, m2_with_2, m1_no_2, m2_no_2, x0_2 = sim_exp2()

# Plot Exp 2 Density Histogram
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(-1, 1, 50)
x_analytical = np.linspace(-0.99, 0.99, 200)
p_analytical_flat = np.ones_like(x_analytical) * 0.5

sns.histplot(x_with, bins=bins, stat='density', label=r'Simulation (With Drift $f_1 = -2x$)', color='#2b5c8f', alpha=0.6, ax=ax)
sns.histplot(x_no, bins=bins, stat='density', label=r'Simulation (Without Drift $f_1 = 0$)', color='#d95f02', alpha=0.4, ax=ax)
ax.plot(x_analytical, p_analytical_flat, 'k--', linewidth=2.5, label=r'Analytical Flat Target $P_{eq}(x) = 1/2$')

ax.set_title(r'Experiment 2: Particle Bounded Between Parallel Walls ($D(x) = 1 - x^2$)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(r'Dimensionless Position $\tilde{x}$')
ax.set_ylabel(r'Probability Density $P(\tilde{x})$')
ax.set_xlim(-1, 1)
ax.set_ylim(0, 1.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(output_dir, 'exp2_bounded_walls.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot Exp 2 Moments
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
m1_target_2_with = x0_2 * np.exp(-2.0 * t2)
m2_target_2_with = (1.0 / 3.0) + (x0_2**2 - 1.0 / 3.0) * np.exp(-6.0 * t2)
m1_target_2_no = np.ones_like(t2) * x0_2
m2_target_2_no = 1.0 + (x0_2**2 - 1.0) * np.exp(-2.0 * t2)

ax1.plot(t2, m1_with_2, 'o', color='#2b5c8f', markersize=3, markevery=15, label=r'Simulation (With Drift $f_1=-2x$)')
ax1.plot(t2, m1_target_2_with, 'k--', linewidth=2, label=r'Analytical With Drift: $x_0 e^{-2t}$')
ax1.plot(t2, m1_no_2, 's', color='#d95f02', markersize=3, markevery=15, label=r'Simulation (Without Drift $f_1=0$)')
ax1.plot(t2, m1_target_2_no, 'r:', linewidth=2, label=r'Analytical No Drift: $x_0$')
ax1.set_title(r'First Moment $\langle \tilde{x}(\tilde{t}) \rangle$', fontsize=13, fontweight='bold')
ax1.set_xlabel(r'Dimensionless Time $\tilde{t}$')
ax1.set_ylabel(r'$\langle \tilde{x} \rangle$')
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

ax2.plot(t2, m2_with_2, 'o', color='#2b5c8f', markersize=3, markevery=15, label=r'Simulation (With Drift $f_1=-2x$)')
ax2.plot(t2, m2_target_2_with, 'k--', linewidth=2, label=r'Analytical Equilibrium Target: $1/3$')
ax2.plot(t2, m2_no_2, 's', color='#d95f02', markersize=3, markevery=15, label=r'Simulation (Without Drift $f_1=0$)')
ax2.plot(t2, m2_target_2_no, 'r:', linewidth=2, label=r'Analytical Unphysical Target: $1.0$')
ax2.set_title(r'Second Moment $\langle \tilde{x}^2(\tilde{t}) \rangle$', fontsize=13, fontweight='bold')
ax2.set_xlabel(r'Dimensionless Time $\tilde{t}$')
ax2.set_ylabel(r'$\langle \tilde{x}^2 \rangle$')
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

fig.suptitle(r'Experiment 2: Moment Relaxation Between Bounded Walls ($D(x) = 1 - x^2$)', fontsize=15, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(output_dir, 'exp2_x_moments.png'), dpi=150, bbox_inches='tight')
plt.close()


# ==============================================================================
# EXPERIMENT 3: Unconfined Particle with Quadratic Diffusion (D(x) = 1 + x^2)
# ==============================================================================
def sim_exp3(N_particles=20000, dt=0.0005, N_steps=800):
    x_with = np.zeros(N_particles)
    x_no   = np.zeros(N_particles)
    
    t_hist, m2_with, m2_no = [], [], []
    sq_dt = np.sqrt(dt)
    t = 0.0
    
    for step in range(N_steps):
        t_hist.append(t)
        m2_with.append(np.mean(x_with**2))
        m2_no.append(np.mean(x_no**2))
        
        dW = np.random.normal(0, sq_dt, N_particles)
        
        # With f1 drift (+2x)
        g_with = np.sqrt(2.0 * (1.0 + x_with**2))
        g_prime_with = 2.0 * x_with / g_with
        dx_with = (2.0 * x_with) * dt + g_with * dW + 0.5 * g_with * g_prime_with * (dW**2 - dt)
        x_with += dx_with
        
        # With zero drift (0)
        g_no = np.sqrt(2.0 * (1.0 + x_no**2))
        g_prime_no = 2.0 * x_no / g_no
        dx_no = 0.0 * dt + g_no * dW + 0.5 * g_no * g_prime_no * (dW**2 - dt)
        x_no += dx_no
        
        t += dt
        
    return np.array(t_hist), np.array(m2_with), np.array(m2_no)

t3, m2_with_3, m2_no_3 = sim_exp3()

# Analytical targets for Exp 3
m2_target_3_with = (1.0 / 3.0) * (np.exp(6.0 * t3) - 1.0)
m2_target_3_no   = np.exp(2.0 * t3) - 1.0

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t3, m2_with_3, 'o', color='#2b5c8f', markersize=4, markevery=20, label=r'Simulation (With Drift $f_1=+2x$)')
ax.plot(t3, m2_target_3_with, 'k--', linewidth=2.5, label=r'Analytical With Drift: $\frac{1}{3}(e^{6\tilde{t}} - 1)$')

ax.plot(t3, m2_no_3, 's', color='#d95f02', markersize=4, markevery=20, label=r'Simulation (Without Drift $f_1=0$)')
ax.plot(t3, m2_target_3_no, 'r:', linewidth=2.5, label=r'Analytical Without Drift: $e^{2\tilde{t}} - 1$')

ax.set_title(r'Experiment 3: Variance Expansion in Quadratic Diffusion ($D(x) = 1 + x^2$)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(r'Dimensionless Time $\tilde{t}$')
ax.set_ylabel(r'Second Moment $\langle \tilde{x}^2 \rangle$')
ax.set_yscale('log')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(output_dir, 'exp3_x_moments.png'), dpi=150, bbox_inches='tight')
plt.close()

print("All simulation and moment plots generated successfully.")
