import math
from scipy.stats import norm
import numpy as np

def risk_neutral_measure (u,d,r,T):
    p = (math.exp (r*T) - d)/(u-d)
    if 0 < p < 1:
        return p
    else:
        raise RuntimeError('Arbitrage on the market!')


def Binomial_price_Am (S: np.array, K: float, r: float, sigma: float, T: float, N: int) -> float:
    """Compute the price of an American put option using binomial model
    S: float: initial price
    K: float: strike price
    r: float: risk-free rate
    sigma: float: volatility
    T: float: maturity
    N: int: time steps
    """
    # Length of one period in the binomial approximation
    dt = T / N
    
    # CRR values of u and d, and the risk neutral measure
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/u
    p = risk_neutral_measure(u, d, r, dt)
    
    # Computation of stock prices in the last period
    idx = np.arange(0, N+1)
    ST = S * np.power(u, idx) * np.power(d, N-idx)
    
    # Payoff at the maturity of the option
    payoff = np.maximum(K - ST, np.zeros_like(ST))
    
    # Iteration backward through the tree
    for period in range(N):
        idx = np.arange (0, N - period)
        ST = S * np.power (u, idx) * np.power (d, N-period-1-idx)
        payoff = np.maximum(K - ST, math.exp(-r*dt)* (p * payoff[1:] + (1-p)* payoff[:-1]))
        
    # The array is of size 1 but we return the value only so that the returned value is of type float 
    # and not type numpy.ndarray
    return payoff[0]


# BS formula for single values of S_0
def BS_European_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes formula for European put
    S: initial stock price
    K: strike
    r: risk-free rate
    sigma: volatility
    T: maturity """

    d1 = (math.log(S/K) + (r + sigma**2/2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)