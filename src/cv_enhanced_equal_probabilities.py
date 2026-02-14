import numpy as np
import math
from scipy.stats import norm

import sys
import os

project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.standard_pricing_formulas import BS_European_put, risk_neutral_measure


def CV_Enhaced_Equal_Probability(S_0: float, K: float, r: float, sigma: float, T: float, N: int) -> float:
    '''
    This function implements the enhanced version of the control 
    variate method for American put options with equal probabilities.
    '''

    # Divide into time steps
    dt = T / N 

    # Compute u, d and the risk-neutral probability measure
    # Under this values for u and d the probability of an up move and a down move are equal to 0.5.
    u = math.exp((r - (sigma**2)/2)*dt + sigma*math.sqrt(dt))
    d = math.exp((r - (sigma**2)/2)*dt - sigma*math.sqrt(dt))
    p = risk_neutral_measure(u, d, r, dt)
    e = math.exp(-r*dt)
    
    # Initialize put_diff
    put_diff = np.zeros(N+1)
    
    # Find j*_n
    JSn = math.floor((math.log(K/(S_0*u**N))/math.log(d/u))) + 1

    for i in range(N - 1, 0, -1):
        
        # At first iteration we use this algorithm to find our first JSi. In the next iteration it will be recycled using JSim1 of this iteration and so on.
        if i == N-1:
            JSi  = JSn

            if JSi > i:
                JSi = i

            # We check if JSi is on an early excercise node.
            # If it is then we try the JSi - 1 until we reach the smallest JSi of the early excercise region
            # otherwise, we go looking for it on higher values of j.
            while JSi >= 0 and e*(p * put_diff[JSi] + (1-p)*put_diff[JSi+1]) <= K - S_0*(u**(i-JSi))*(d**JSi) - BS_European_put(S_0*(u**(i-JSi))*(d**JSi), K, r, sigma, (T - i*dt)):
                
                JSi -= 1 

            JSi += 1 

            while JSi <= i and e*(p * put_diff[JSi] + (1-p)*put_diff[JSi+1]) > K - S_0*(u**(i-JSi))*(d**JSi) - BS_European_put(S_0*(u**(i-JSi))*(d**JSi), K, r, sigma, (T - i*dt)):
                
                JSi += 1

        # We recycle the value of JSim1 from the previuous iteration of i.
        else:
            JSi = JSim1

        if JSi == 0:
            # If JSi is equal to zero then all the previous nodes will be early excercise. 
            # Therefore the put value will be given by its payoff at time 0.
            return float(K - S_0)

        # We update the values of put_diff until JSi - 1 by normal backward induction.
        put_diff[:JSi] = e*(p*put_diff[:JSi] + (1-p)*put_diff[1:JSi + 1]) 
        
        # We start looking for JSim1
        JSim1 = JSi - 1

        if JSim1 > i-1 :
            JSim1 = i - 1

        # We update the early excercise value at time i at the index JSi.
        put_diff[JSi] = K - S_0*u**(i - JSi)*d**JSi - BS_European_put(S_0*u**(i - JSi)*d**JSi, K, r, sigma, (T - i*dt))

        # While we are in the early exercise region at i - 1 we decrease JSim1.
        while JSim1 >= 0 and e*(p*put_diff[JSim1] + (1-p)*put_diff[JSim1+1]) <= K - S_0*u**(i-1 - JSim1)*d**JSim1 - BS_European_put(S_0*u**(i-1 - JSim1)*d**JSim1, K, r, sigma, (T - (i-1)*dt)):

            JSim1 -= 1
        
        JSim1 += 1

        # We update the early excercise value at time i the index JSi + 1.
        put_diff[JSi+1] = K - S_0*u**(i - JSi - 1)*d**(JSi + 1) - BS_European_put(S_0*u**(i - JSi - 1)*d**(JSi + 1), K, r, sigma, (T - i*dt))

        # While we are in the continuation region at i - 1  we increase JSim1 and simultaneusly we update the early exercise nodes at time i.
        while JSim1 <= i - 1 and e*(p*put_diff[JSim1] + (1-p)*put_diff[JSim1+1]) > K - S_0*u**(i-1 - JSim1)*d**JSim1 - BS_European_put(S_0*u**(i-1 - JSim1)*d**JSim1, K, r, sigma, (T - (i-1)*dt)):
            
            JSim1 += 1
            put_diff[JSim1+1] = K - S_0*u**(i - JSim1 - 1)*d**(JSim1 + 1) - BS_European_put(S_0*u**(i - JSim1 - 1)*d**(JSim1 + 1), K, r, sigma, (T - i*dt))


    # Last step for i = 0.
    put_diff = e*(p*put_diff[0] + (1-p)*put_diff[1])

    return float(max((put_diff + BS_European_put(S_0, K, r, sigma, T)), K - S_0)) 