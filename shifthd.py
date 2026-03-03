import numpy as np
import pandas as pd
from scipy.stats.mstats import hdquantiles
import matplotlib.pyplot as plt

# https://github.com/GRousselet/blog/blob/master/shift_function/wilcox_modified.txt

def shifthd(x, y, nboot=200):
    """
    Compute confidence intervals for the difference between deciles
    of two independent groups using the Harrell-Davis estimator.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Remove missing values (equivalent to x<-x[!is.na(x)] in R)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    nx = len(x)
    ny = len(y)
    
    # Calculate the critical value
    crit = 80.1 / (min(nx, ny)**2) + 2.73
    
    # List to hold our rows of results before turning into a DataFrame
    results = []
    
    # Loop through deciles 1 to 9 (0.1 to 0.9)
    for d in range(1, 10):
        q = d / 10.0
        
        # --- Bootstrap for X ---
        # Sample with replacement to create a matrix of shape (nboot, nx)
        boot_data_x = np.random.choice(x, size=(nboot, nx), replace=True)
        # Apply the Harrell-Davis estimator to each row
        bvec_x = [float(hdquantiles(row, prob=q)[0]) for row in boot_data_x]
        # Calculate sample variance (ddof=1 matches R's default var())
        sex = np.var(bvec_x, ddof=1)
        
        # --- Bootstrap for Y ---
        # Sample with replacement to create a matrix of shape (nboot, ny)
        boot_data_y = np.random.choice(y, size=(nboot, ny), replace=True)
        # Apply the Harrell-Davis estimator to each row
        bvec_y = [float(hdquantiles(row, prob=q)[0]) for row in boot_data_y]
        # Calculate sample variance
        sey = np.var(bvec_y, ddof=1)
        
        # --- Calculate Metrics ---
        # Base Harrell-Davis estimates for the original x and y at quantile q
        hd_x = float(hdquantiles(x, prob=q)[0])
        hd_y = float(hdquantiles(y, prob=q)[0])
        
        difference = hd_x - hd_y
        margin_of_error = crit * np.sqrt(sex + sey)
        
        ci_lower = difference - margin_of_error
        ci_upper = difference + margin_of_error
        
        # Store the row
        results.append({
            'decile': q,
            'group1': hd_x,
            'group2': hd_y,
            'difference': difference,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    # Return as a pandas DataFrame (equivalent to data.frame() in R)
    # This makes it very easy to plot using libraries like Seaborn/Plotnine
    out = pd.DataFrame(results)
    
    # Setting the decile as the index makes it cleaner, though optional
    out.set_index('decile', inplace=True)
    
    return out


def plot(out):
  yerr = out['ci_upper'] - out['difference']
  plt.errorbar(out.index, out['difference'], yerr=yerr, fmt='-o', capsize=5)
  plt.grid(True)

