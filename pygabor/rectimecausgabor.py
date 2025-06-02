"""pygabor.rectimecausgabor 
"""

import numpy as np
from matplotlib import pyplot as plt


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
    L : int

    def __init__(self, omega, delta_t = 1.0, c = 2.0, numlevels = 16, N = 1.0, L = 100):

        # Distribution parameter
        self.c = c

        assert self.c > 1

        # Number of temporal scale levels
        self.K = numlevels

        assert self.K > 1

        # Proportionality factor
        self.N = N

        assert self.N >= 1

        # Angular frequencies [rad/s]
        self.omega = np.array(omega)

        self.J = np.size(self.omega)

        assert self.J > 1

        self.frequencies = self.omega / (2*np.pi)

        # Initialize the output buffers
        self.level = np.zeros((self.J, self.K, 2))
        self.level_prev = np.zeros((self.J, self.K, 2))

        # Sampling time [s]
        self.delta_t = delta_t

        assert self.delta_t > 0

        # Sampling rate [Hz]
        r = 1.0/delta_t

        # Initialize the time constants
        self.mu = np.zeros((self.J, self.K))
        self.gain = np.zeros((self.J, self.K))

        for j in range(self.J):
            sigma_j0 = 2*np.pi*N/omega[j]
            tau_jref = np.pow(r * sigma_j0, 2)

            tau_jk_1 = 0
            for k in range(self.K):
                # Compute the temporal scale levels
                tau_jk = np.pow(c, 2*(k-self.K)) * tau_jref

                # Compute the temporal scale increments
                delta_tau_jk = tau_jk - tau_jk_1
                tau_jk_1 = tau_jk

                # Compute the time constant
                self.mu[j,k] = 0.5*(np.sqrt(1 + 4 * delta_tau_jk) - 1)

                # Set the gain
                self.gain[j,k] = 1/(1+self.mu[j,k])

        self.n = 0

        # Signal length
        self.L = L

        assert self.L > 1

        self.duration = self.L * self.delta_t

        self.spectrogramchart = np.zeros((self.J*self.K, self.L))
    
    def process(self, signal):
        for j in range(self.J):
            t = self.n * self.delta_t
            omega_jt = self.omega[j] * t
            input = np.array([signal * np.cos(omega_jt), -signal * np.sin(omega_jt)])

            aux = input
            for k in range(self.K):
                self.level[j,k,:] = self.level_prev[j,k,:] + self.gain[j,k] * (aux - self.level_prev[j,k,:])
                aux = self.level[j,k,:]
        
        self.level_prev = self.level.copy()

        self.n += 1

        return self.level

    def spectrogram(self, lowsoftthresh : float = 0.000001, maxrange : float = 60):

        # Compute the absolute value of the spectrogram
        absspectrogram = np.sqrt(self.level[:,:,0]**2 + self.level[:,:,1]**2)

        # Compute logarithmic magnitudes in dB, with additional lower bound
        maxval = np.max(absspectrogram)
        logspectrogram = 20 * np.log10(absspectrogram/maxval + lowsoftthresh)
        logspectrogram[logspectrogram < -maxrange] = -maxrange

        self.spectrogramchart[:,self.n-1] = logspectrogram.flatten()

        im = \
        plt.imshow(self.spectrogramchart, \
                    cmap='jet', interpolation='nearest', aspect='auto', \
                    origin='lower', \
                    extent=[0, self.duration, min(self.frequencies), max(self.frequencies)])
        plt.colorbar(im)

        plt.xlabel("Time (seconds)")
        plt.ylabel("Log Frequency (Hz)")
        plt.show(block=False)

        return logspectrogram