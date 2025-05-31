"""pygabor.rectimecausgabor 
"""

import numpy as np


class RecursiveTimeCausGaborMethod:
    omega : np.ndarray
    mu :  np.ndarray
    delta_t : float
    c : float
    N : float
    J : int
    K : int
    level : np.ndarray
    level_prev : np.ndarray
    n : int

    def __init__(self, omega, delta_t = 1.0, c = 2.0, numlevels = 8, N = 1.0):

        # Distribution parameter
        self.c = c

        assert c > 1

        # Number of temporal scale levels
        self.K = numlevels

        assert numlevels > 1

        # Proportionality factor
        self.N = N

        assert N >= 1

        # Angular frequencies [rad/s]
        self.omega = omega

        self.J = np.len(omega)       

        assert self.J > 1

        # Initialize the output buffers
        self.level = np.zeros((J, K, 2))
        self.level_prev = np.zeros((J, K, 2))

        # Sampling time [s]
        self.delta_t = delta_t

        assert delta_t > 0

        # Sampling rate [Hz]
        r = 1.0/delta_t

        # Initialize the time constants
        self.mu = np.zeros((J, K))
        self.gain = np.zeros((J, K))

        for j in range(J):
            sigma_j0 = 2*np.pi*N/omega[j]
            tau_jref = np.pow(r * sigma_j0, 2)

            tau_jk_1 = 0
            for k in range(K):
                # Compute the temporal scale levels
                tau_jk = np.pow(c, 2*k-K) * tau_jref

                # Compute the temporal scale increments
                delta_tau_jk = tau_jk - tau_jk_1
                tau_jk_1 = tau_jk

                # Compute the time constant
                self.mu[j,k] = 0.5*(np.sqrt(1 + 4 * delta_tau_jk) - 1)

                # Set the gain
                self.gain[j,k] = 1/(1+self.mu[j,k])

        self.n = 0
    
    def process(signal):
        for j in range(J):
            t = n * delta_t
            omega_jt = omega[j] * t
            input = np.array([signal * np.cos(omega_jt), -signal * np.sin(omega_jt)])

            aux = input
            for k in range(K):
                level[j,k,:] = aux = level_prev[j,k,:] + gain[j,k] * (aux - level_prev[j,k,:])
        
        level_prev = level

        n += 1

        return level
