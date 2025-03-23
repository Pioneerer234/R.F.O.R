# R.F.O.R
This script simulates the motion of a particle in a rotating reference frame using Newton's second law with fictitious forces. It takes into account:  Coriolis force:  𝐹 𝐶 = − 2 𝑚 ( 𝛺 × 𝑣 ) F  C ​  =−2m(Ω×v)  Centrifugal force:  𝐹 cf = − 𝑚 𝛺 × ( 𝛺 × 𝑟 ) F  cf ​  =−mΩ×(Ω×r)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
m = 1.0  # Mass of the particle
Omega = np.array([0, 0, 1])  # Angular velocity (along z-axis)
omega_mag = np.linalg.norm(Omega)  # Magnitude of angular velocity

# Initial conditions
r0 = np.array([1.0, 0.0, 0.0])  # Initial position (meters)
v0 = np.array([0.0, 1.0, 0.0])  # Initial velocity (m/s)

# Time span
t_span = (0, 10)  # Simulate for 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Time points for evaluation

def equations_of_motion(t, y):
    """Computes the derivatives of position and velocity in the rotating frame."""
    r = y[:3]  # Position vector
    v = y[3:]  # Velocity vector

    # Coriolis force
    F_coriolis = -2 * m * np.cross(Omega, v)

    # Centrifugal force
    F_centrifugal = -m * np.cross(Omega, np.cross(Omega, r))

    # Total fictitious force
    F_fictitious = F_coriolis + F_centrifugal

    # Acceleration (Newton's 2nd Law)
    a = F_fictitious / m  # Since F = ma

    return np.hstack((v, a))  # Return derivatives [dr/dt, dv/dt]

# Solve the system using Runge-Kutta (RK45)
y0 = np.hstack((r0, v0))  # Initial state vector [r, v]
solution = solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval, method='RK45')

# Extract results
t_vals = solution.t
r_vals = solution.y[:3, :]

# Plotting the trajectory in the rotating frame
plt.figure(figsize=(8, 6))
plt.plot(r_vals[0], r_vals[1], label="Trajectory in rotating frame", color='b')
plt.scatter(r0[0], r0[1], color='r', label="Starting Point")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Particle Motion in a Rotating Frame")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
