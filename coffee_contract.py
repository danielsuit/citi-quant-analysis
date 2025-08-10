import numpy as np
from scipy.stats import norm
S_t = 1.20  # Spot price in dollars
S_0 = 1.20  # Spot price in dollars
X = 1.25  # Strike price in dollars
r = 0.02  # Risk-free rate
d = 0.01  # Storage cost
T = 0.5  # Time to maturity in years
sigma = 0.25  # Volatility (25%)
# Cost of Carry
F_t = S_t * np.exp((r + d) * T)
print(f"The fair price of the coffee futures contract is ${F_t:.3f} per pound.")
# Black-Scholes
d1 = (np.log(S_0 / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
C = S_0 * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
print(f"The price of the call option is ${C:.3f}.")
# Monte Carlo Simulation
num_simulations = 10000
num_steps = 252
dt = T / num_steps
np.random.seed(42)
price_paths = np.zeros((num_steps, num_simulations))
price_paths[0] = S_0
for t in range(1, num_steps):
    z = np.random.standard_normal(num_simulations)
    price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
# Calculating the average simulated price at maturity
average_price = np.mean(price_paths[-1])
variance = np.var(price_paths[-1])
print(f"The variance of the simulated prices is {variance:.3f}.")
print(f"The average simulated price using Monte Carlo Simulation ${average_price:.3f}.")

