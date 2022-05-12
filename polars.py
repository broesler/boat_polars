#!/usr/bin/env python3
# =============================================================================
#     File: polars.py
#  Created: 2022-05-11 21:21
#   Author: Bernie Roesler
#
"""
Description: Plot polars and minimize time to travel for a given TWA/TWS.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds

π = np.pi

datafile = Path('./prospector.csv')
df = pd.read_csv(datafile, delimiter=';')
df = (df.rename({'twa/tws':'TWA'}, axis=1)
        .assign(TWA_deg=lambda x: x['TWA'],
                TWA_rad=lambda x: x['TWA_deg'] * π/180)
        .set_index('TWA_rad')
        .drop('TWA', axis=1)
        .sort_index()
        .replace(0, np.nan)
        )
df.columns.name = 'TWS'

# Choose true wind spped [kts]
TWS = '12'
assert TWS in df.columns

# Fit a spline to the data
tf = df[TWS].dropna()
tf.at[0] = 0
tf.at[π] = 0

θ = np.linspace(tf.index.min(), tf.index.max())
V = interp1d(tf.index, tf, kind='cubic')

# Define the objective function
L = 10  # [nm] distance to end point
α = 120 * π/180  # [rad] true wind angle


class TrianglePath():
    """The triangle path object."""

    def __init__(self, θ, d, α=0, V=V, L=L):
        """
        Parameters
        ----------
        θ : float
            Starting angle.
        d : float
            Distance along starting angle. Must be ≤ L.
        α : float
            The true wind angle.
        V : callable, optional
            Velocity as a function of `θ`.
        """
        self.θ = θ
        self.d = d
        self.α = α
        self.V = V
        self.d_1 = (d**2 + L**2 - 2*d*L*np.cos(θ))**0.5  # law of cosines
        self.θ_1 = np.arccos((L - d*np.cos(θ)) / self.d_1)

    @property
    def total_distance(self):
        return self.d + self.d_1

    @property
    def time_0(self):
        return self.d / self.V(self.α - self.θ) 

    @property
    def time_1(self):
        return self.d_1 / self.V(self.α - self.θ_1) 

    @property
    def total_time(self):
        return self.time_0 + self.time_1

    @property
    def average_velocity(self):
        return self.total_distance / self.total_time

    # TODO define get/set methods to keep all consistent


# Define the objective function separately for fast evaluation
def time_to_point(θ, d, α=0, V=V, L=L):
    """Compute the time to traverse the triangle.

    Parameters
    ----------
    θ : float
        Starting angle.
    d : float
        Distance along starting angle. Must be ≤ L.
    α : float
        The true wind angle.
    V : callable, optional
        Velocity as a function of `θ`.

    Returns
    -------
    time : float
        The time in [hr] to traverse the edges of the triangle.
    """
    d_1 = (d**2 + L**2 - 2*d*L*np.cos(θ))**0.5  # law of cosines
    θ_1 = np.arccos((L - d*np.cos(θ)) / d_1)
    return d / V(α - θ) + d_1 / V(α - θ_1)


# TODO 
#   * return d_1, total distance, average velocity
#   * add cost of tack/gybe (function of turning angle)

res = minimize(lambda x: time_to_point(*x, α=α, V=V, L=L),
               x0=np.r_[1, 1],
               bounds=((0, π/2), (0, L)),
               )

tp = TrianglePath(*res.x, α=α, L=L, V=V)

print(f"Average speed is {tp.average_velocity - V(α):.2f} [kts] greater than straight-line.")
print(f"Time is {tp.total_time - L/V(α):.2f} [hr] faster than straight-line.")

# ----------------------------------------------------------------------------- 
#         Plots
# -----------------------------------------------------------------------------
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.axvline(0, c='C3', ls='--', lw=1)
ax.plot((θ - α) * 180/π, V(θ), 'C0-', zorder=1)
ax.plot((tf.index - α) * 180/π, tf.values, 'C0x', label=rf"U = {TWS}")
ax.scatter(tp.θ * 180/π, V(α + tp.θ), c='C3', marker='o')
ax.scatter(-tp.θ_1 * 180/π, V(α - tp.θ_1), c='C3', marker='o')
ax.set(xlabel='Relative wind angle [deg]',
       ylabel='Velocity [kts]')
ax.grid(True)
ax.legend()

plt.show()

# =============================================================================
# =============================================================================
