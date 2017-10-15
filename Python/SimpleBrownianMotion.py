from bokeh.plotting import figure, show
import random
import numpy as np

class BrownianMotion:

    def __init__(self, num_vectors, num_points):
        self._vectors = num_vectors
        self._points = num_points
        self._std_normal = np.random.standard_normal((self._vectors, self._points)) # generates standard normals
        self._A = np.triu(np.ones((self._points, self._points))) # gets upper triangular matrix
        self._brownian_motion = np.dot(self._std_normal, self._A) # calculates brownian motion

    def get_brownian_motion(self):
        return self._brownian_motion

    def get_covariance(self):
        return np.cov(self._A.transpose(), self._A)

def plot(bm, title=''):
    '''
    This function plots 1000 paths of a given process
    '''

    # Randomly generate 1000 colors for the 1000 paths
    r = lambda: random.randint(0, 255)
    color = ['#%02X%02X%02X' % (r(), r(), r()) for _ in range(1000)]

    # Generate x-values for the 1000 paths
    x_axis = [np.arange(0, 1, 1 / 100)
              for _ in range(1000)]

    # Plot the graph using Bokeh multi_line
    # Note that all the data points have to be in
    # list objects not Numpy Arrays
    fig = figure(plot_width=800, plot_height=400, title=title)
    fig.multi_line(x_axis, list(bm), color=color)
    show(fig)

if __name__ == "__main__":
    # Create a BrownianMotion object and get the realization
    brownian_motion = BrownianMotion(1000, 100).get_brownian_motion()

    plot(brownian_motion, 'Brownian Motion')

    # Brownian Bridge

    brownian_motion_time_adjusted = brownian_motion/10

    brownian_bridge = np.zeros((1000, 100))

    for i in range(100):
        brownian_bridge[:, i] = brownian_motion_time_adjusted[:, i] - (i+1)/100 * brownian_motion_time_adjusted[:, 99]

    plot(brownian_bridge, 'Brownian Bridge')



