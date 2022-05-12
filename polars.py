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

from scipy.interpolate import interp1d
from pathlib import Path

π = np.pi

datafile = Path('./prospector.csv')
df = pd.read_csv(datafile, sep='\s+;\s+')
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

θ = np.linspace(tf.index.min(), tf.index.max())
V = interp1d(tf.index, tf, kind='cubic')

# ----------------------------------------------------------------------------- 
#         Plots
# -----------------------------------------------------------------------------
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.plot(θ * 180/π, V(θ), 'C0-')
ax.plot(tf.index * 180/π, tf.values, 'C0x')
ax.grid(True)

plt.show()

# =============================================================================
# =============================================================================
