import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formulas with dividend yield
def bs_call_price(S, K, T, r, sigma, q):
    """
    Returns the Black-Scholes call option price with dividend yield.
    """
    if sigma <= 0 or T <= 0:
        return max(0, np.exp(-q * T) * S - np.exp(-r * T) * K)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def bs_put_price(S, K, T, r, sigma, q):
    """
    Returns the Black-Scholes put option price with dividend yield.
    """
    if sigma <= 0 or T <= 0:
        return max(0, np.exp(-r * T) * K - np.exp(-q * T) * S)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put

# Intrinsic value functions
def intrinsic_call(S, K):
    return np.maximum(S - K, 0)

def intrinsic_put(S, K):
    return np.maximum(K - S, 0)

# Streamlit App
def main():
    # Top control area
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strike = st.slider("Strike:", min_value=10, max_value=200, value=100, step=5)
            time_to_maturity = st.slider("Time to maturity:", 
                                                 min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        
        with col2:
            risk_free_rate = st.slider("Risk-free rate:", 
                                               min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100.0
            volatility = st.slider("Volatility:", 
                                           min_value=1.0, max_value=100.0, value=20.0, step=1.0) / 100.0
        
        with col3:
            dividend_yield = st.slider("Dividend yield:", 
                                               min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
        

    

    # Underlying price range
    S_min = 0
    S_max = 2 * strike
    S_vals = np.linspace(S_min, S_max, 200)
    
    # Compute option prices and intrinsic values
    call_vals = [bs_call_price(S, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield) for S in S_vals]
    put_vals = [bs_put_price(S, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield) for S in S_vals]
    intrinsic_call_vals = intrinsic_call(S_vals, strike)
    intrinsic_put_vals = intrinsic_put(S_vals, strike)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Call option plot
    ax1.plot(S_vals, call_vals, label="Call Value", color="blue")
    ax1.plot(S_vals, intrinsic_call_vals, label="Intrinsic Value", linestyle="--", color="gray")
    ax1.set_xlabel("Underlying Price (S)")
    ax1.set_ylabel("Option Value")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Put option plot
    ax2.plot(S_vals, put_vals, label="Put Value", color="red")
    ax2.plot(S_vals, intrinsic_put_vals, label="Intrinsic Value", linestyle="--", color="gray")
    ax2.set_xlabel("Underlying Price (S)")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    # Display plots
    st.pyplot(fig)

if __name__ == "__main__":
    main()
