import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import sympy as sym

def evaluate_expression(expression, current_val_stoch_process):
    '''
    This function evaluates the SymPy expression

    You want to make sure that you're working with a SymPy object
    when evaluating the expression.

    So you have to take some precaution by casting the current type
    to a SymPy type
    '''

    x = sym.Symbol('x')

    if type(expression) == int:
        expression = sym.Integer(expression)

    if type(expression) == float:
        expression = sym.Float(expression)

    f = sym.lambdify(x, expression, "numpy")

    return f(current_val_stoch_process)


def milstein_scheme(drift, sigma, sigma_x, Z, t0 = 0, T = 1, N = 100, X0 = 1.0, paths=1000):
    '''
    This function approximates the solution to an SDE
    using the Milstein scheme
    '''

    if Z.shape[0] != paths:
        paths = Z.shape[0]

    if Z.shape[1] != N:
        N = Z.shape[1]

    Y = np.zeros((paths, N+1))
    Y[:, 0] = X0

    delta = 1 / N
    sqrt_delta = np.sqrt(delta)

    delta_wiener = np.zeros((paths, N+1))

    for i in range(1, N+1):
        b = evaluate_expression(drift, Y[:, i-1])
        sig = evaluate_expression(sigma, Y[:, i-1])
        sig_x = evaluate_expression(sigma_x, Y[:, i-1])

        delta_wiener[:, i] = sqrt_delta * Z[:, i-1]

        Y[:, i] = Y[:, i-1] + b * delta + sig * delta_wiener[:, i] + \
                  0.5 * sig * sig_x * (delta_wiener[:, i] * delta_wiener[:, i] - delta)

    return Y

def euler_scheme(drift, sigma, Z, t0 = 0, T = 1, N = 100, X0 = 1.0, paths=1000):
    '''
    This function approximates the solution to an SDE using the
    Euler scheme
    '''

    if Z.shape[0] != paths:
        paths = Z.shape[0]

    if Z.shape[1] != N:
        N = Z.shape[1]

    Y = np.zeros((paths, N + 1))
    Y[:, 0] = X0

    delta = 1 / N
    sqrt_delta = np.sqrt(delta)

    delta_wiener = np.zeros((paths, N + 1))

    for i in range(1, N + 1):
        b = evaluate_expression(drift, Y[:, i - 1])
        sig = evaluate_expression(sigma, Y[:, i - 1])

        delta_wiener[:, i] = sqrt_delta * Z[:, i - 1]

        Y[:, i] = Y[:, i - 1] + b * delta + sig * delta_wiener[:, i]

    return Y

def plot(approx, method):
    '''
    This function plots the approximation
    '''

    r = lambda: np.random.randint(0, 255)

    colors = ['#%02X%02X%02X' % (r(), r(), r()) for _ in range(approx.shape[0])]

    x_axis = [index for index in range(approx.shape[1])]

    multiple_axis = [x_axis for _ in range(approx.shape[0])]

    title = 'Approximating SDE using ' + method + ' scheme'

    sde = figure(plot_width=800, plot_height=400, title=title)

    sde.multi_line(multiple_axis, list(approx), color=colors)

    show(sde)

def initialize_stochastic_differential_equation():
    '''
    This function initializes the stochastic differential equation
    '''

    x = sym.Symbol('x') # need this to use SymPy

    drift = sym.sympify(input("Enter drift function: ").strip())

    sigma = sym.sympify(input("Enter volatility function: ").strip())

    sigma_x = sym.diff(sigma, x)

    paths = int(input("How many paths? ").strip())

    N = int(input("How many points? ").strip())

    Z = np.random.standard_normal((paths, N))

    return drift, sigma, sigma_x, Z

def error(euler, milstein):
    e = abs(milstein - euler)

    error_plot = figure(plot_width=800, plot_height=400, title='Milstein and Euler Schemes')
    x_axis = [[np.log2(x) for x in range(1, len(e[0])+1)] for _ in range(len(e))]
    r = lambda: np.random.randint(0, 255)
    colors = ['#%02X%02X%02X' % (r(), r(), r()) for _ in range(len(e))]
    # error_plot.multi_line(x_axis, list(e), color=colors)
    error_plot.multi_line(x_axis, list(euler), legend='Euler')
    error_plot.multi_line(x_axis, list(milstein), line_dash="4 4", color=colors, legend='Milstein')
    absolute_error = figure(plot_width=800, plot_height=400, title='Absolute Error between Milstein and Euler')
    absolute_error.multi_line(x_axis, list(e), color=colors)

    g = gridplot([error_plot, absolute_error], ncols=1)

    show(g)

if __name__ == '__main__':

    x = sym.Symbol('x') # need this to use SymPy

    drift, sigma, sigma_x, Z = initialize_stochastic_differential_equation()

    milstein = milstein_scheme(drift, sigma, sigma_x, Z=Z, X0=1)

    # plot(milstein, 'Milstein')

    euler = euler_scheme(drift, sigma, Z=Z, X0=1)

    # plot(euler, 'Euler')

    error(euler, milstein)


