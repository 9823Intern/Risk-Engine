import pybind11
import pandas as pd
import numpy as np
from scipy import stats


class VaR:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def calculate_VaR(self, confidence_level: float = 0.99, method: str = 'parametric', 
                     data_type: str = 'returns') -> pd.Series:
        """
        Calculate Value at Risk (VaR) for given confidence level.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95, 0.99)
            method: 'parametric' (normal distribution) or 'historical'
            data_type: 'returns' (percentage returns) or 'pnl' (profit/loss)
        
        Returns:
            pd.Series: VaR for each strategy/column
        """
        if method == 'parametric':
            return self._parametric_var(confidence_level, data_type)
        elif method == 'historical':
            return self._historical_var(confidence_level)
        else:
            raise ValueError("Method must be 'parametric' or 'historical'")
    
    def _parametric_var(self, confidence_level: float, data_type: str) -> pd.Series:
        """Calculate parametric VaR assuming normal distribution."""
        # Get z-score for confidence level (left tail)
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        
        # Calculate mean and std for each strategy
        mean_returns = self.data.mean()
        std_returns = self.data.std()
        
        if data_type == 'returns':
            # For returns data: VaR = -(mean + z_score * std)
            # Negative because VaR is typically reported as positive loss
            var = -(mean_returns + z_score * std_returns)
        elif data_type == 'pnl':
            # For PnL data: VaR = mean + z_score * std  
            # z_score is negative, so this gives us the loss at confidence level
            var = mean_returns + z_score * std_returns
        else:
            raise ValueError("data_type must be 'returns' or 'pnl'")
        
        return var
    
    def _historical_var(self, confidence_level: float) -> pd.Series:
        """Calculate historical VaR using empirical quantiles."""
        alpha = 1 - confidence_level
        
        # For each strategy, find the quantile corresponding to the confidence level
        var_values = {}
        for column in self.data.columns:
            column_data = self.data[column].dropna()
            if len(column_data) == 0:
                var_values[column] = np.nan
            else:
                # Historical VaR is the quantile at alpha level
                var_values[column] = -column_data.quantile(alpha)  # Negative for loss
        
        return pd.Series(var_values)
    
    def calculate_conditional_var(self, confidence_level: float = 0.99, 
                                 method: str = 'parametric', data_type: str = 'returns') -> pd.Series:
        """
        Calculate Conditional VaR (Expected Shortfall) - average loss beyond VaR.
        """
        if method == 'historical':
            return self._historical_cvar(confidence_level)
        elif method == 'parametric':
            return self._parametric_cvar(confidence_level, data_type)
        else:
            raise ValueError("Method must be 'parametric' or 'historical'")
    
    def _parametric_cvar(self, confidence_level: float, data_type: str) -> pd.Series:
        """Calculate parametric Conditional VaR assuming normal distribution."""
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        
        mean_returns = self.data.mean()
        std_returns = self.data.std()
        
        # Expected value below VaR threshold for normal distribution
        expected_shortfall_factor = stats.norm.pdf(z_score) / alpha
        
        if data_type == 'returns':
            cvar = -(mean_returns + std_returns * expected_shortfall_factor)
        elif data_type == 'pnl':
            cvar = mean_returns + std_returns * expected_shortfall_factor
        else:
            raise ValueError("data_type must be 'returns' or 'pnl'")
        
        return cvar
    
    def _historical_cvar(self, confidence_level: float) -> pd.Series:
        """Calculate historical Conditional VaR using empirical data."""
        alpha = 1 - confidence_level
        
        cvar_values = {}
        for column in self.data.columns:
            column_data = self.data[column].dropna()
            if len(column_data) == 0:
                cvar_values[column] = np.nan
            else:
                # CVaR is the mean of all values at or below the VaR threshold
                var_threshold = column_data.quantile(alpha)
                tail_losses = column_data[column_data <= var_threshold]
                cvar_values[column] = -tail_losses.mean()  # Negative for loss
        
        return pd.Series(cvar_values)