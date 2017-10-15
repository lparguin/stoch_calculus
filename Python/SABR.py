import numpy as np
from bokeh.plotting import show, figure
from bokeh.palettes import RdYlBu
from bokeh.models import Legend

std_normals = np.random.standard_normal((100, 100))
covariance = np.cov(std_normals)
correlated_normals = np.dot(np.linalg.cholesky(covariance), std_normals)

def sigma(alpha, sigma0, correlated_normal):
    stoch_vol = np.zeros((len(correlated_normal), len(correlated_normal[0])))
    dWt = np.zeros((len(correlated_normal), len(correlated_normal[0])))
    stoch_vol[:, 0] = sigma0
    sqrt_delta = np.sqrt(1 / len(correlated_normal[0]))
    for i in range(1, len(correlated_normal[0])):
        dWt[:, i] = correlated_normal[:, i-1] * sqrt_delta
        stoch_vol[:, i] = stoch_vol[:, i-1] + alpha * stoch_vol[:, i-1] * dWt[:, i]
    return stoch_vol

alpha = float(input('Enter an alpha (>= 0): ').strip())
sigma0 = float(input('Enter an initial volatility: ').strip())

corr_normal_for_stoch_vol = correlated_normals[:correlated_normals.shape[0]//2]
corr_normal_for_forward_rate = correlated_normals[correlated_normals.shape[0]//2:]

stoch_vol = sigma(alpha, sigma0, corr_normal_for_stoch_vol)

x_axis = [[x for x in range(correlated_normals.shape[1])]
          for _ in range(len(corr_normal_for_stoch_vol))]

stoch_vol_plot_title = 'Stochastic Volatility with alpha = {} and initial volatility = {}'.format(alpha, sigma0)
stoch_vol_plot = figure(plot_width=800, plot_height=400, title=stoch_vol_plot_title)
stoch_vol_plot.multi_line(x_axis, list(stoch_vol))

show(stoch_vol_plot)

def forward_rate(stoch_vol, beta, f0, correlated_normal):
    f = np.zeros((len(correlated_normal), len(correlated_normal[0])))
    dWt = np.zeros((len(correlated_normal), len(correlated_normal[0])))
    f[:, 0] = f0
    sqrt_delta = np.sqrt(1 / len(correlated_normal[0]))
    for i in range(1, len(correlated_normal[0])):
        dWt[:, i] = correlated_normal[:, i] * sqrt_delta
        f[:, i] = f[:, i-1] + stoch_vol[:, i] * np.power(f[:, i-1], beta) * dWt[:, i]

    return f

libor_6_month = 1.48
f = [forward_rate(stoch_vol, beta, libor_6_month, corr_normal_for_forward_rate)
     for beta in np.arange(0, 1.1, 0.1)]

r = lambda: np.random.randint(0, 255)
colors = ["#%02X%02X%02X" % (r(), r(), r()) for _ in range(len(corr_normal_for_forward_rate))]

forward_rate_path_title = 'SABR Model with beta ranging from 0.0 to 1.0 and initial rate = {}'.format(libor_6_month)
forward_rate_paths = figure(plot_width=800, plot_height=400, title=forward_rate_path_title)

forward_rate_beta_half = figure(plot_width=800, plot_height=400, title='Beta = 0.5')
forward_rate_beta_half.multi_line(x_axis, list(f[len(f)//2]), color=colors)

show(forward_rate_beta_half)

for path, color, beta in zip(f, RdYlBu[len(f)], np.arange(0, 1.1, 0.1)):
    forward_rate_paths.line(x_axis[0], list(path[0]), color=color, legend=str(beta))

forward_rate_paths.legend.location = "top_left"
forward_rate_paths.legend.orientation = "horizontal"
show(forward_rate_paths)
