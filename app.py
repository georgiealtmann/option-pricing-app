# importing necessary libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from numpy import exp, sqrt, log

# creating class (I will call another one with different model): Black-Scholes Model 
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
    # initialize model parameters
        self.T = time_to_maturity
        self.K = strike
        self.S = current_price 
        self.sigma = volatility 
        self.r = interest_rate # risk-free interest rate

    def run(self):
        T, K, S, sigma, r = self.T, self.K, self.S, self.sigma, self.r
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        self.call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2) # calculate European call option price
        self.call_delta = norm.cdf(d1) # calculate Delta for call

# Barrier option via Monte Carlo class
class BarrierOptionMonteCarlo:
    def __init__(self, S0, K, H, T, r, sigma, n_simulations=10000, n_steps=2520):
        self.S0 = S0 # initial spot price
        self.K = K  # strike price
        self.H = H # barrier level
        self.T = T # time to maturity
        self.r = r # risk-free interest rate
        self.sigma = sigma # volatility
        self.n_simulations = n_simulations # number of Monte Carlo simulations (M)
        self.n_steps = n_steps # number of time steps in each simulation path (N), can be adjusted for trading days in Y 

    def simulate(self):
        dt = self.T / self.n_steps
        # drift and diffusion terms for GBM based on literature
        drift = (self.r - 0.5 * self.sigma**2) * dt 
        diffusion = self.sigma * np.sqrt(dt) 
        # generate all random normal shocks
        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        # initialize asset price paths matrix
        S_paths = np.zeros((self.n_simulations, self.n_steps + 1))
        S_paths[:, 0] = self.S0
        # simulate paths using geometric Brownian motion
        for t in range(1, self.n_steps + 1):
            S_paths[:, t] = S_paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])

        return S_paths

    def price(self):
        S_paths = self.simulate() # simulate price paths
        S_T = S_paths[:, -1] # get final prices at maturity
        barrier_crossed = np.any(S_paths >= self.H, axis=1)  # check for each path whether it ever crossed the barrier
        valid_paths = ~barrier_crossed # valid if barrier not hit

        # payoffs for only call and zeroed if barrier was breached
        payoffs = np.maximum(S_T - self.K, 0)  
        payoffs[~valid_paths] = 0.0 
        discounted_payoff = np.exp(-self.r * self.T) * payoffs
        return np.mean(discounted_payoff), S_paths

# --- Streamlit Layout and Model Selector ---
st.set_page_config(page_title="Option Pricing", layout="centered")
st.title("Option Pricing Models")

# Sidebar for option selection
option_mode = st.sidebar.radio("Choose Option Type:", ["Vanilla European Call Option", "Exotic (Barrier Call) Option"])

#  sidebar and description for vanilla European call option 
if option_mode == "Vanilla European Call Option":
    st.markdown("This calculator prices **European-style call options** using the Black-Scholes model.")

    S = st.sidebar.number_input("Spot Price (S)", value=100.0, key="S_vanilla")
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, key="K_vanilla")
    T = st.sidebar.slider("Time to Maturity (T)", 0.10, 30.0, 1.0, step=0.1, key="T_vanilla")
    sigma_percent = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0, step=0.5, key="sigma_vanilla")
    r_percent = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 20.0, 5.0, step=0.1, key="r_vanilla")

    sigma = sigma_percent / 100
    r = r_percent / 100

    # run Black-Scholes model
    model = BlackScholes(T, K, S, sigma, r)
    model.run()

    # disolay output: price
    st.subheader("Call Option Price")
    st.metric("Call Option Price", f"{model.call_price:.2f}")

    st.subheader("Call Delta vs Spot Price")
    spot_prices = np.linspace(S * 0.5, S * 1.5, 100) # generate 100 evenly spaced spot prices ranging from 50% to 150% of the current spot price S
    deltas = []

    # for each spot price, calculate delta using Black-Scholes d1 formula
    for s in spot_prices:
        d1 = (log(s / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        deltas.append(norm.cdf(d1))

     # plot the delta curve
    fig, ax = plt.subplots()
    ax.plot(spot_prices, deltas)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Call Delta")
    ax.set_title("Call Option Delta Across Spot Prices")
    ax.grid(True)
    st.pyplot(fig)

    #this part follows the same logic as above
elif option_mode == "Exotic (Barrier Call) Option":
    st.markdown("This calculator prices **up-and-out barrier call options** using Monte Carlo simulation.")
    
    st.sidebar.caption("💡 Recommended (only locally!): M = 50,000 simulations | N = 2,520 time steps")

    S_ex = st.sidebar.number_input("Spot Price (S)", value=100.0, key="S_ex")
    K_ex = st.sidebar.number_input("Strike Price (K)", value=100.0, key="K_ex")
    H_ex = st.sidebar.number_input("Barrier Level (H)", value=130.0, key="H_ex")
    T_ex = st.sidebar.slider("Time to Maturity (T)", 0.10, 30.0, 1.0, step=0.1, key="T_ex")
    sigma_ex_percent = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0, step=0.5, key="sigma_ex")
    r_ex_percent = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 20.0, 5.0, step=0.1, key="r_ex")
    n_sim = st.sidebar.number_input("Number of Simulations (M)", value=10000, step=1000, key="n_sim")
    n_steps = st.sidebar.number_input("Time Steps (N)", value=252, step=50, key="n_steps")

    sigma_ex = sigma_ex_percent / 100
    r_ex = r_ex_percent / 100

    st.subheader("Barrier Call Option Price")
    with st.spinner("Running simulation..."):
        start_time = time.time()
        barrier_model = BarrierOptionMonteCarlo(S0=S_ex, K=K_ex, H=H_ex, T=T_ex, r=r_ex,
                                                sigma=sigma_ex, n_simulations=n_sim, n_steps=n_steps)
        price_exotic, paths = barrier_model.price()
        elapsed = time.time() - start_time

    st.write(f"Monte Carlo estimated price (up-and-out call): **{price_exotic:.2f}**")

    st.subheader("Sample Forecasted Brownian Paths")
    fig2, ax2 = plt.subplots()
    for i in range(min(50, n_sim)):
        ax2.plot(paths[i], lw=0.5)
    ax2.axhline(H_ex, color='red', linestyle='--', label='Barrier Level H')
    ax2.set_title("Simulated Asset Paths with Barrier Level")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Spot Price")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Option Premium vs Spot Price")
    spot_range = np.linspace(S_ex * 0.8, H_ex * 1.1, 50) # generates 50 evenly spaced prices from 80% of the initial spot price (S_ex) to 110% of the barrier level (H_ex)
    prices = []

    for spot in spot_range:
        bm = BarrierOptionMonteCarlo(S0=spot, K=K_ex, H=H_ex, T=T_ex, r=r_ex, sigma=sigma_ex,
                                     n_simulations=3000, n_steps=n_steps)
        price, _ = bm.price()
        prices.append(price)

    fig3, ax3 = plt.subplots()
    ax3.plot(spot_range, prices, label="Call Option Premium", color="blue")
    ax3.axvline(H_ex, color="red", linestyle="--", label="Barrier Level H")
    ax3.set_xlabel("Spot Price")
    ax3.set_ylabel("Call Option Premium")
    ax3.set_title("Option Premium vs Spot Price")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
