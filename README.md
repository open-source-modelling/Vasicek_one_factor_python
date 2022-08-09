<h1 align="center" style="border-botom: none">
  <b>
    🐍 Vasicek One-Factor model 🐍     
  </b>
</h1>

Vasicek one factor model for simulating the evolution of a credit instruments such as a government bonds. The Vasicek model assumes that the process evolves as an Ornstein-Uhlenbeck process. Ornstein-Uhlenbeck is a stochastic process where over time, the process tends to drift towards a long-term mean (mean reverting). 

## Problem
When trying to simulate for example credit spreads, there is a variety of models. The choice of the model and its limitations are a key factor in deciding which model to implement. There are compeling economic arguments in favor of mean reversion.  

## Solution
One of the simplest models, the [Vasicek one factor model](https://en.wikipedia.org/wiki/Vasicek_model) assumes that the credit market can be described by a simple mean reverting stochastic process with one source of uncertainty comming from a [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion). One limitation is that due to to normal distribution of the noise, the process allows negative spreads which might be undesirable in certain circumstances.

The stochastic differential equation (SDE) of the Vasicek model is shown on the Wiki page https://en.wikipedia.org/wiki/Vasicek_model 

### Input

  - `r0`    ... float, starting interest rate of the Vasicek process
  - `a`     ... float, speed of reversion" parameter that characterizes the velocity at which such trajectories will regroup around b in time
  - `lam`   ... float, long term mean level. All future trajectories of r will evolve around this mean level in the long run
  - `sigma` ... float, instantaneous volatility measures instant by instant the amplitude of randomness entering the system
  - `T`     ... integer, end modelling time. From 0 to T the time series runs.
  - `dt`    ... float, increment of time that the proces runs on. Ex. dt = 0.1 then the time series is 0, 0.1, 0.2,...

### Output

 - `interest_rate_simulation` N x 2 DataFrame with a sample path as values and modelling time as index

## Getting started
```python
import numpy as np
import pandas as pd
from Vasicek_one_factor import simulate_Vasicek_One_Factor

r0 = 0.1 # The starting interest rate
a = 1.0 # speed of reversion parameter
lam = 0.1 # long-term mean interest rate level correction
sigma = 0.2 # instantaneous volatility
T = 52 # end modelling time
dt = 0.1 # increments of time

print(simulate_Vasicek_One_Factor(r0, a, lam, sigma, T, dt))
```
