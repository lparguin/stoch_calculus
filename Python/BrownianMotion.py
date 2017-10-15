import numpy as np
from bokeh.plotting import show, figure
import random
import math

'''
This program is generates a Brownian Motion object
and should be used in other files when Brownian Motion
is involved.
'''

class BrownianMotion:

    def __init__(self, num_vectors, num_points):
        # We want to generate our own vector of IID Normals
        self._iid_std_normal = np.random.standard_normal((num_vectors, num_points))
        self._num_vectors = num_vectors
        self._num_points = num_points

        # By default,
        self.decomposition_bm()

        self._bm_type = 'Decomposition'


    def shape(self):
        '''
        Returns the shape of the matrix
        '''

        return (self._num_vectors, self._num_points)

    def realization(self):
        '''
        Returns the data points for the
        realization of the Brownian Motion
        '''

        return self._brownian_motion

    def set_brownian_motion(self, bm, type):
        '''
        Upon updating the method of implementation
        for Brownian Motion, we update the Brownian Motion variable
        and the type of Brownian motion.
        '''

        self._brownian_motion = bm
        self._bm_type = type

    def wavelet_bm(self):
        '''
        Function returns a realization of Brownian Motion using
        Wavelet functions
        '''

        def delta(t):
            '''
            Returns the delta portion of the formula
            '''

            if t >= 0 and t < 0.5:
                return 2*t
            elif t>= 0.5 and t <= 1.0:
                return 2*(1-t)
            else:
                return 0

        def get_j_k(n):
            '''
            Solves for j and k in n = 2^j + k
            given n as a paramter
            '''

            if n != 0:
                if n - np.power(2, round(np.log10(n) / np.log10(2), 0)) < 0:
                    j = math.floor(np.log10(n) / np.log10(2))
                else:
                    j = round(np.log10(n) / np.log10(2), 0)
            else:
                j = 0

            k = math.ceil(n-(np.power(2, j)))

            return j, k

        bm = np.zeros((self._num_vectors, self._num_points))

        for t in np.arange(0, 1, 1 / self._num_points):
            xt = np.zeros(self._num_vectors)
            for n in range(0, self._num_points):
                j, k = get_j_k(n)
                _lambda = 0.5 * np.power(2, -j / 2) if n != 0 else 1
                _delta = delta(np.power(2, j) * t - k) if n != 0 else t
                xt[:] += self._iid_std_normal[:, n] * _lambda * _delta
            bm[:, int(t * self._num_points)] = xt


        self.set_brownian_motion(bm, 'Wavelet')

        return bm


    def karhunen_loeve_bm(self):
        '''
        Function returns a realization of Brownian Motion using
        Karhunen-Loeve Expansion
        '''

        def phi(t, T=1):
            '''
            Returns for all n, phi(t)
            (Formula in Numerical Solution of SDE through
            Computer Experiments)
            '''

            phi_arr = np.zeros(self._num_points)
            for n in range(0, self._num_points):
                phi_arr[n] = (2 * np.sqrt(2 * T) / (2 * n * np.pi + np.pi) * np.sin((2 * n * np.pi * t + np.pi * t) / (2 * T)))
            return phi_arr


        phi_collection = [phi(t) for t in np.arange(0, 1, 1/ self._num_points)]

        bm = np.zeros((self._num_vectors, self._num_points))

        for i in range(self._num_vectors):
            n = self._iid_std_normal[i, :] * phi_collection
            bm[i] = [np.sum(n[j]) for j in range(len(n))]

        self.set_brownian_motion(bm, 'Karhunen-Loeve')

        return bm


    def decomposition_bm(self):
        '''
        Generates Brownian Motion using decomposition
        '''

        A = np.triu(np.ones((self._num_points, self._num_points)))
        bm = np.dot(self._iid_std_normal, A) / 10

        self.set_brownian_motion(bm, 'Decomposition')

        return bm

    def linear_interpolation_bm(self):
        '''
        Generates linearly interpolated Brownian Motion
        '''

        bm = np.zeros((self._num_vectors, self._num_points))
        delta = 1 / self._num_points
        sqrt_delta = np.sqrt(delta)
        tk = -delta

        x_axis = np.zeros((self._num_vectors, self._num_points))
        for i in range(1, self._num_points):
            tk += delta
            x_axis[:, i] = tk
            bm[:, i] = bm[:, i-1] + self._iid_std_normal[:, i] * sqrt_delta

        self.set_brownian_motion(bm, 'Linearly Interpolated')

        return bm

    def plot(self, plot_width=600, plot_height=300):
        '''
        Plots the realization of Brownian Motion
        '''

        color = [color for color in color_generator(self._num_vectors)]
        x_axis = [x_axis for x_axis in x_axis_generator(self._num_points, self._num_vectors)]
        fig = figure(plot_width=plot_width, plot_height=plot_height, title=self._bm_type)
        fig.multi_line(x_axis, list(self._brownian_motion), color=color)
        show(fig)

    def var(self, t=None):
        '''
        Solves for variance of the process at a point in time t
        or just general variance of the entire process
        '''

        if t == None:
            sample_var = np.var(self._brownian_motion)
        else:
            index = int(t * self._num_points) - 1
            sample_var = np.var(self._brownian_motion[:, index])
        return sample_var

def x_axis_generator(num_points, num_vectors):
    '''
    Generator function to give x_axis range
    '''

    for _ in range(num_vectors):
        yield np.arange(0, 1, 1 / num_points)

def color_generator(num_vectors):
    '''
    Generator function to give random color
    '''

    r = lambda: random.randint(0, 255)
    for _ in range(num_vectors):
        yield '#%02X%02X%02X' % (r(), r(), r())

def cov(bm1, bm2):
    sample_cov = np.cov(bm1, bm2)
    return sample_cov

if __name__ == '__main__':
    test_bm = BrownianMotion(1000, 100)

    test_bm.wavelet_bm()

    test_bm.plot()








