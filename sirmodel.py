import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import warnings
import mplcursors


# # # # # # # # # # # # # # #  Initial Conditions # # # # # # # # # # # # # # # 
   
N = 763 # Number of boys in the school
D = 2 # Number of days infected boys can infect susceptible population
alpha = 1/D # Rate of recovery
beta = 1.75 # Expected amount of people an infected person infects per day.


I0 = 1 # First carrier of disease is the only one infected at day = 0
S0 = N - I0 # Remaining population is susceptible at day = 0
R0 = 0 # No one in the population has recovered at day = 0


# # # # # # # # # # # # # # #  Data Values from BMJ # # # # # # # # # # # # # # # 

true_x = np.arange(15)
true_y = [1, 3, 8, 25, 75, 227, 296, 258, 236, 192, 126, 71, 28, 11, 7]

# # # # # # # # # # # #  Calculating and Plotting SIR Graphs # # # # # # # # # # # # 

def deriv(y, t, N, beta, alpha): # using the SIR formulae
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - alpha * I
    dRdt = alpha * I
    return dSdt, dIdt, dRdt


t = np.arange(0, 14.25, 0.25) # Grid of time points (in days)
y0 = S0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, alpha))
S, I, R = ret.T
s, i, r = S/N, I/N, R/N


def sir_graphs(t, s, i, r, x, y):
    f, ax = plt.subplots()
    ax.plot(t, s, 'b', linewidth=2, label='Susceptible')
    ax.plot(t, r, 'g', linewidth=2, label='Recovered')
    ax.plot(t, i, 'r', linewidth=2, label='Infected')

    ax.scatter(x, y, marker='x', label='Data Points',  s=15, c='black')
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    ax.set_xlabel('Time in Days')
    ax.set_ylabel('Proportion of Population')
    ax.set_title('SIR Model of the Influenza Epidemic in 1978')
    ax.grid(False)
    
    l = ax.legend()
    for spine in ('top', 'right'):
      ax.spines[spine].set_visible(False)
    mplcursors.cursor(hover = True)
    plt.show()


sir_graphs(t, s, i, r, true_x, [i/N for i in true_y])

warnings.filterwarnings("ignore", category=DeprecationWarning) 
