import numpy as np

class GaussianVector:

    def __init__(self, mean=[], cov=[]):
        try:
            if len(mean) == len(cov) and len(mean) == len(cov[0]):
                self._mean = mean
                self._cov = cov
            else:
                raise("Invalid Dimensions")
        except:
            '''Allow the default to be standard Gaussian'''
            self._mean = np.zeros(len(mean))
            self._cov = np.identity(len(mean))


    def generate_random_vector(self):
        '''
        This function returns a Gaussian vector
        given some mean vector and covariance matrix

        Args:
            mean: an array-like structure containing means
            cov: a covariance matrix

        Returns:
            Numpy array with n Gaussian-distributed random numbers
        '''
        return np.random.multivariate_normal(self._mean, self._cov)

    def get_covariance_matrix(self):
        '''
        This function returns the covariance matrix from which we
        constructed the Gaussian vector

        Returns:
            Covariance Matrix
        '''
        return self._cov

    def get_mean_vector(self):
        '''
        This function returns the mean vector from which we
        constructed the Gaussian vector

        Returns:
            Mean vector
        '''
        return self._mean

if __name__ == '__main__':
    mean1 = np.zeros(2)
    cov1 = [[2, 1],[1, 2]]
    gv1 = GaussianVector(mean1, cov1)

    print( gv1.get_covariance_matrix() )
    print( gv1.get_mean_vector() )
    print( gv1.generate_random_vector() )

    mean2 = np.zeros(5)
    cov2 = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    gv2 = GaussianVector(mean2, cov2)

    print( gv2.get_covariance_matrix() )
    print( gv2.get_mean_vector() )
    print( gv2.generate_random_vector() )
