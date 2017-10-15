import numpy as np
from scipy.stats import norm
from bokeh.plotting import figure, show

def graph_option(option, title):
    fig = figure(plot_width=600, plot_height=300,
                 title=title)

    fig.line(np.arange(0, len(option)), option)

    show(fig)

def d1(t, x, K, T):
    delta_t = T - t
    sqrt_delta_t = np.sqrt(delta_t)

    return 1 / sqrt_delta_t * (np.log(x / K) + 0.5 * delta_t)

def d2(t, x, K, T):
    delta_t = T - t
    sqrt_delta_t = np.sqrt(delta_t)

    return 1 / sqrt_delta_t * (np.log(x / K) - 0.5 * delta_t)

def call_option_price(t, x, K, T):
    return x*norm.cdf(d1(t, x, K, T)) - norm.cdf(d2(t, x, K, T))*K

if __name__ == '__main__':
    
    K = 100 # strike price

    call_option_payoff = [x - K if x > K else 0 for x in range(0, 201)]

    graph_option(call_option_payoff, 'Underlying Price vs. Option Payoff')

    # Obvious error when  x = 0
    call_option_prices = [call_option_price(0.999, x, 100, 1) for x in range(1, 201)]

    graph_option(call_option_prices, 'Underlying Price vs. Option Price')

