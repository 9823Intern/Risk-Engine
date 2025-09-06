import pandas as pd
import numpy as np
from VaR import VaR

# Create sample data similar to what you described
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='D')
strategies = ['Strategy_A', 'Strategy_B', 'Strategy_C']

# Sample returns data (daily percentage returns)
returns_data = pd.DataFrame({
    'Strategy_A': np.random.normal(0.0008, 0.02, 252),  # 0.08% daily mean, 2% daily vol
    'Strategy_B': np.random.normal(0.0005, 0.015, 252), # 0.05% daily mean, 1.5% daily vol  
    'Strategy_C': np.random.normal(0.001, 0.025, 252)   # 0.1% daily mean, 2.5% daily vol
}, index=dates)

# Sample PnL data (daily profit/loss in dollars)
pnl_data = pd.DataFrame({
    'Strategy_A': np.random.normal(800, 20000, 252),   # $800 daily mean, $20k daily vol
    'Strategy_B': np.random.normal(500, 15000, 252),   # $500 daily mean, $15k daily vol
    'Strategy_C': np.random.normal(1000, 25000, 252)   # $1000 daily mean, $25k daily vol
}, index=dates)

print("=== VaR Analysis Results ===\n")

# Test with returns data
print("1. Returns Data Analysis:")
print(f"Sample returns data shape: {returns_data.shape}")
print(f"Date range: {returns_data.index[0].date()} to {returns_data.index[-1].date()}\n")

var_calculator_returns = VaR(returns_data)

# Calculate VaR with different confidence levels and methods
confidence_levels = [0.95, 0.99]
methods = ['parametric', 'historical']

for confidence in confidence_levels:
    print(f"--- {confidence*100}% Confidence Level ---")
    for method in methods:
        var_results = var_calculator_returns.calculate_VaR(
            confidence_level=confidence, 
            method=method, 
            data_type='returns'
        )
        print(f"{method.capitalize()} VaR:")
        for strategy, var_value in var_results.items():
            print(f"  {strategy}: {var_value:.4f} ({var_value*100:.2f}%)")
        
        # Also calculate Conditional VaR
        cvar_results = var_calculator_returns.calculate_conditional_var(
            confidence_level=confidence,
            method=method,
            data_type='returns'
        )
        print(f"{method.capitalize()} CVaR (Expected Shortfall):")
        for strategy, cvar_value in cvar_results.items():
            print(f"  {strategy}: {cvar_value:.4f} ({cvar_value*100:.2f}%)")
        print()

print("\n" + "="*50 + "\n")

# Test with PnL data
print("2. PnL Data Analysis:")
print(f"Sample PnL data shape: {pnl_data.shape}")
print(f"Date range: {pnl_data.index[0].date()} to {pnl_data.index[-1].date()}\n")

var_calculator_pnl = VaR(pnl_data)

for confidence in confidence_levels:
    print(f"--- {confidence*100}% Confidence Level ---")
    for method in methods:
        var_results = var_calculator_pnl.calculate_VaR(
            confidence_level=confidence,
            method=method,
            data_type='pnl'
        )
        print(f"{method.capitalize()} VaR:")
        for strategy, var_value in var_results.items():
            print(f"  {strategy}: ${var_value:,.2f}")
        
        cvar_results = var_calculator_pnl.calculate_conditional_var(
            confidence_level=confidence,
            method=method,
            data_type='pnl'
        )
        print(f"{method.capitalize()} CVaR (Expected Shortfall):")
        for strategy, cvar_value in cvar_results.items():
            print(f"  {strategy}: ${cvar_value:,.2f}")
        print()

print("\n=== Usage Example for Your CSV Data ===")
print("""
# To use with your actual CSV file:

# Load your data
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Create VaR calculator
var_calc = VaR(df)

# For returns data (daily percentage returns):
var_99 = var_calc.calculate_VaR(confidence_level=0.99, method='parametric', data_type='returns')
cvar_99 = var_calc.calculate_conditional_var(confidence_level=0.99, method='parametric', data_type='returns')

# For PnL data (daily profit/loss):
var_99_pnl = var_calc.calculate_VaR(confidence_level=0.99, method='parametric', data_type='pnl')
cvar_99_pnl = var_calc.calculate_conditional_var(confidence_level=0.99, method='parametric', data_type='pnl')

print("99% VaR:", var_99)
print("99% CVaR:", cvar_99)
""") 