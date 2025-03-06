# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# endregion

# region problem c: Hydraulic Valve System
def ode_system(t, X, A, Cd, ps, pa, V, beta, rho, Kvalve, m, y):
    """
    Defines the system of ODEs for the hydraulic valve system.
    """
    x, xdot, p1, p2 = X
    xddot = (p1 - p2) * A / m
    p1dot = (y * Kvalve * (ps - p1) - rho * A * xdot) * beta / (rho * V)
    p2dot = (-y * Kvalve * (p2 - pa) - rho * A * xdot) * beta / (rho * V)
    return [xdot, xddot, p1dot, p2dot]


def main():
    t_span = (0, 0.02)
    t_eval = np.linspace(0, 0.02, 200)
    params = (4.909E-4, 0.6, 1.4E7, 1.0E5, 1.473E-4, 2.0E9, 850.0, 2.0E-5, 30, 0.002)
    pa = 1.0E5
    ic = [0, 0, pa, pa]
    sln = solve_ivp(ode_system, t_span, ic, args=params, t_eval=t_eval)

    plt.subplot(2, 1, 1)
    plt.plot(t_eval, sln.y[1], 'b-', label='Velocity')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_eval, sln.y[2], 'r-', label='P1')
    plt.plot(t_eval, sln.y[3], 'b-', label='P2')
    plt.ylabel('Pressure (Pa)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.show()


if __name__ == "__main__":
    main()
