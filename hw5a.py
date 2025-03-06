# region imports
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# endregion

# region problem a: Moody Diagram
def ff(Re, rr, CBEQN=False):
    """
    Calculates the Darcy-Weisbach friction factor.
    Uses the Colebrook equation for turbulent flow and 64/Re for laminar flow.
    """
    if CBEQN:
        cb = lambda f: 1 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))
        result = fsolve(cb, 0.02)
        return result[0]
    else:
        return 64 / Re


def plotMoody():
    """
    Generates the Moody Diagram with various relative roughness values.
    """
    ReValsCB = np.logspace(np.log10(4000), np.log10(1e8), 20)
    ReValsL = np.logspace(np.log10(600), np.log10(2000), 20)
    rrVals = np.logspace(-6, -1.3, 20)

    ffLam = np.array([ff(Re, 0) for Re in ReValsL])
    ffCB = np.array([[ff(Re, rr, CBEQN=True) for Re in ReValsCB] for rr in rrVals])

    plt.loglog(ReValsL, ffLam, 'k-', label='Laminar Flow')
    for nRelR in range(len(rrVals)):
        plt.loglog(ReValsCB, ffCB[nRelR], color='k')
    plt.xlabel("Reynolds Number")
    plt.ylabel("Friction Factor")
    plt.grid(True, which='both', linestyle='--')
    plt.show()


def main():
    plotMoody()


if __name__ == "__main__":
    main()

