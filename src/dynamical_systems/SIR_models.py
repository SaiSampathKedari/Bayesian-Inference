import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple

def sir_dynamics_identifiable(t : float, 
                     x: np.ndarray, 
                     theta_dyn: np.ndarray):
    """
    Identifiable SIR dynamics: x = [S, I, R], theta_dyn = [beta, r, delta]
    """
    
    N = 1000.0
    S, I, R = x
    beta, r, delta = theta_dyn
    
    dS = delta * N - delta * S - beta * S * I
    dI = beta * S * I - (r + delta) * I
    dR = r * I - delta * R
    
    return np.array([dS, dI, dR])

def sir_dynamics_nonIdentifiable(t : float,
                                 x: np.ndarray,
                                 theta_dyn: np.ndarray ):
    """
    Non-Identifiable SIR dynamics: x = [S, I, R], theta_dyn = [gamma, k, r, delta]
    """
    N = 1000.0
    S, I, R = x
    gamma, k, r, delta = theta_dyn
    
    dS = delta * N - delta * S - gamma * k * S * I
    dI = gamma * k * S * I - (r + delta) * I
    dR = r * I - delta * R
    
    return np.array([dS, dI, dR])
    
def sir_observation_model(x_traj: np.ndarray) -> float:
    """
    Observation model h(x, theta_obs). For this dynamics: observe I(t).
    x_traj: shape (3, T)
    returns: shape (T,)
    """
    return x_traj[1,:]


def solve_SIR_forward(theta_dyn : np.ndarray, 
                      SIR_dynamics: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                      t_eval : np.ndarray,
                      x0: np.ndarray = np.array([900.0, 100.0, 0.0])
                      ):
    """
    Solve SIR ODE forward for given parameters
    """

    SIR_dynamics_wrapped = lambda t, x : SIR_dynamics(t, x, theta_dyn)
    
    sol = solve_ivp(SIR_dynamics_wrapped,
                    (t_eval[0], t_eval[-1]), 
                    y0=x0, 
                    method='RK45', 
                    t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    
    return sol

def log_likelihood_SIR( theta_dyn: np.ndarray, 
                        y_obs: np.ndarray, 
                        std: float,
                        t_eval: np.ndarray,
                        solve_SIR_fun : Callable[[float, np.ndarray, np.ndarray], np.ndarray]):
    
    """
    Log-likelihood for observed infected counts
    """
        
    sol = solve_SIR_forward(theta_dyn, 
                            solve_SIR_fun,
                            t_eval)
    y_pred = sir_observation_model(sol.y)
    diff =  y_pred - y_obs
    out = -0.5*np.sum(np.square(diff))/ std**2
    return out


def log_prior_gaussian(theta : np.ndarray, 
                  mean : np.ndarray, 
                  cov: np.ndarray):
    """
    Multivariate Gaussian log-prior
    """
    
    diff = theta - mean
    inv_cov = np.linalg.inv(cov)
    out = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
    return out

def log_posterior_SIR(theta_dyn : np.ndarray, 
                      y_obs : np.ndarray, 
                      std : float,
                      t_eval: np.ndarray,
                      solve_SIR_fun : Callable[[float, np.ndarray, np.ndarray], np.ndarray], 
                      prior_mean, 
                      prior_cov):
    """
    Log-posterior for SIR parameter inference
    """
    ll = log_likelihood_SIR(theta_dyn, y_obs, std, t_eval, solve_SIR_fun)
    lp = log_prior_gaussian(theta_dyn, prior_mean, prior_cov)
    
    return ll + lp


def laplace_approx(initial_guess, logpost):
    """Perform the laplace approximation.

    Return the MAP point and an approximation of the covariance

    Inputs
    ------
    initial_guess: (nparam, ) array of initial parameters
    logpost: function (param) -> log posterior

    Ouputs
    ------
    map_point: (nparam, ) MAP of the posterior
    cov_approx: (nparam, nparam), covariance matrix for Gaussian fit at MAP
    """
    def neg_post(x):
        """Negative posteror because optimizer is a minimizer"""
        return -logpost(x)

    # Gradient free method to obtain optimum
    res = scipy.optimize.minimize(neg_post, initial_guess, method='Nelder-Mead')
    # Gradient method which also approximates the inverse of the hessian
    res = scipy.optimize.minimize(neg_post, res.x)

    map_point = res.x
    cov_approx = res.hess_inv
    return map_point, cov_approx


def plot_autocorrelation_3d(
    lag_values: np.ndarray,
    acf_values: np.ndarray,
    dim_names: Tuple[str, str, str] = ("dim 1", "dim 2", "dim 3"),
    figsize: Tuple[int, int] = (15, 4),
    marker: str = "o",
    save_path: Optional[str] = None
):
    """
    Visualize autocorrelation function (ACF) for a 3-dimensional chain.

    Parameters
    ----------
    lag_values : np.ndarray
        Lags at which ACF is evaluated. Shape (L,).

    acf_values : np.ndarray
        Autocorrelation values. Shape (L, 3).

    dim_names : tuple of str, optional
        Titles for the three dimensions plotted side by side.
    """

    if acf_values.shape[1] != 3:
        raise ValueError("This visualization function is only for 3D samples.")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, ax in enumerate(axes):
        ax.plot(lag_values, acf_values[:, i], marker=marker, linestyle="-")
        ax.set_title(f"Autocorrelation: {dim_names[i]}", fontsize=14)
        ax.set_xlabel("Lag", fontsize=12)
        ax.set_ylabel("ACF", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=1, alpha=0.8)
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    
    
    