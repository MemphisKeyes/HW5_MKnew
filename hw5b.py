# region imports
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# Define fallback functions in case moody_diagram is unavailable
def ff(Re, rr, CBEQN=False):
    """
    Calculates the Darcy-Weisbach friction factor.
    Uses the Colebrook equation for turbulent flow and 64/Re for laminar flow.
    """
    from scipy.optimize import fsolve
    if CBEQN:
        cb = lambda f: 1/np.sqrt(f) + 2.0 * np.log10(rr/3.7 + 2.51/(Re * np.sqrt(f)))
        result = fsolve(cb, 0.02)
        return result[0]
    else:
        return 64/Re

def plotMoody(ax):
    """
    Generates a basic Moody Diagram in case the module is missing.
    """
    ReVals = np.logspace(2, 8, 50)
    ffVals = [ff(Re, 0.002, CBEQN=True) for Re in ReVals]
    ax.loglog(ReVals, ffVals, 'k-', label='Moody Approximation')
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Friction Factor")
    ax.grid(True, which='both', linestyle='--')
    ax.legend()

def plotPoint(Re, f):
    """
    Plots a point on the Moody diagram in the same figure.
    """
    fig, ax = plt.subplots()
    plotMoody(ax)
    ax.plot(Re, f, 'ro', markersize=8, markeredgecolor='red', markerfacecolor='none')
    plt.show()

# endregion

# region problem b: User Input & Moody Diagram
def ffPoint(Re, rr):
    """
    Determines friction factor based on flow regime, interpolates in transition region.
    """
    if Re >= 4000:
        return ff(Re, rr, CBEQN=True)
    if Re <= 2000:
        return ff(Re, rr)
    CBff = ff(Re, rr, CBEQN=True)
    Lamff = ff(Re, rr)
    mean = (CBff + Lamff) / 2
    sig = 0.2 * mean
    return rnd.gauss(mean, sig)

def main():
    Re = 5000  # Simulated user input
    rr = 0.002  # Simulated user input
    f = ffPoint(Re, rr)
    plotPoint(Re, f)

if __name__ == "__main__":
    main()
