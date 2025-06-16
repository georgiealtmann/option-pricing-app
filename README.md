# Option Pricing Models App using Streamlit

This interactive Python app implements two option pricing models and allows users to explore their behavior under different market conditions. 

## Live Application

[Launch the App](https://option-pricing-app-ermqchngxhk4blubgtb2hz.streamlit.app/)

## Features

### 1. Vanilla Options – Black-Scholes Model
- Computes European-style call option prices using the closed-form Black-Scholes formula
- Visualizes the delta across a range of spot prices
- Fully interactive parameter controls for spot price, volatility, maturity, and rik-free rate

### 2. Exotic Options – Up-and-Out Barrier Call (Monte Carlo)
- Uses simulated geometric Brownian motion to price barrier options
- Implements barrier breach logic with payoff cancellation
- Displays forecasted Brownian paths
- Plots how the premium decays as the spot price approaches the barrier
- Adjustable number of simulations (M) and time steps (N)  
  **Recommended settings:** M = 50,000 | N = 2,520

## Technology Stack

- Python
- Streamlit
- NumPy
- SciPy
- Matplotlib

## Running Locally

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
