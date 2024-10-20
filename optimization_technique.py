# -*- coding: utf-8 -*-
"""
@author: Abhay Sachan
        CC24MTECH11004
"""

import pandas as pd
import numpy as np
import scipy.optimize as optimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as pit

# Load excel data
data = pd.read_excel("C:\\Users\\abhay\\py4e\\CC24MTECH11004\\Autocatalytic_Rxn_1.xlsx")
time = data.iloc[:, 0]
CA = data.iloc[:, 1]
CB = data.iloc[:, 2]
CC = data.iloc[:, 3]

# defining ode
def odes(t, y, k1, k2):
    CA, CB, CC = y
    dCAdt = -k1 * CA * CB
    dCBdt = (k1 * CA * CB) - (k2 * CB)
    dCCdt = k2 * CB
    return [dCAdt, dCBdt, dCCdt]

# initial conditions
x0 = [CA[0], CB[0], CC[0]]

# Defining the Objective function 'fun' for fitting
# 'fun' computes the difference between observed and simulated data. Using this to optimize k1 and k2
def fun(params):
    k1, k2 = params
    soln = solve_ivp(odes, (time.min(), time.max()), y0=x0, args=(k1, k2), t_eval=time)
    CAsol, CBsol, CCsol = soln.y
    difference = np.sum((CAsol - CA)**2 + (CBsol - CB)**2 + (CCsol - CC)**2)
    return difference

# 'mthd' variable to choose different methods of optimization
# Initial guess for k1 and k2 = 0.1, 0.1
# Minimize the 'fun' function
mthd=int(input('Choose method for optimization \n 1. SLSQP Method \n 2. Powell Method \n 3. Nedler-Mead Method \n 4. L-BFGS-B Method \n Enter your chocice 1, 2, 3, 4: '))
bnds=((0, None), (0, None))
if mthd == 1:
    print('\nSolving by SLSQP optimization method: \n')
    result = optimize.minimize(fun, (0.1, 0.1), method='SLSQP', bounds=bnds)
elif mthd == 2:
    print('\nSolving by Powell optimization method: \n')    
    result = optimize.minimize(fun, (0.1, 0.1), method='Powell', bounds=bnds)
elif mthd == 3:
    print('\nSolving by Nedler-Mead optimization method: \n')    
    result = optimize.minimize(fun, (0.1, 0.1), method='Nelder-Mead', bounds=bnds)   
elif mthd == 4:    
    print('\nSolving by L-BFGS-B optimization method: \n')    
    result = optimize.minimize(fun, (0.1, 0.1), method='L-BFGS-B', bounds=bnds)   
else:
    print('\nInvalid choice solving by default with L-BFGS-B optimization method\n')
    result = optimize.minimize(fun, (0.1, 0.1), method='L-BFGS-B', bounds=bnds) 
  
k1_opt, k2_opt = result.x

print('Optimal value of k1: ',k1_opt)
print('Optimal value of k2: ',k2_opt)

# Using the optimal k1 and k2 values to solve the odes again to get CB as a function of time
soln_opt = solve_ivp(odes, (time.min(), time.max()), y0=x0, args=(k1_opt, k2_opt), t_eval=time)
CA_opt, CB_opt, CC_opt = soln_opt.y

# Finding the time and value at which CB is maximized
# Find index of maximum CB
max_index = np.argmax(CB_opt)
max_time = time[max_index]
max_CB = CB_opt[max_index]

print('Time at which CB maximizes: ',max_time)
print('Maximum value of CB: ',max_CB)

# Calculating Root Mean Squared Error (RMSE)
def calculate_overall_rmse(CA, CB, CC, CA_opt, CB_opt, CC_opt):
    # Calculate RMSE for each species
    RMSE_CA = np.sqrt(np.mean((CA - CA_opt) ** 2))
    RMSE_CB = np.sqrt(np.mean((CB - CB_opt) ** 2))
    RMSE_CC = np.sqrt(np.mean((CC - CC_opt) ** 2))
    
    # Calculate overall RMSE by averaging the individual RMSEs
    overall_rmse = np.mean([RMSE_CA, RMSE_CB, RMSE_CC])
    
    return overall_rmse

overall_rmse = calculate_overall_rmse(CA, CB, CC, CA_opt, CB_opt, CC_opt)

# Printing the overall RMSE
print('Root Mean Squared Error :', overall_rmse)

# Plotting actual vs modeled data with different colors
pit.figure(figsize=(10, 6))
pit.plot(time, CA, 'g-', label='Actual CA', markersize=5)       # Actual CA in green (solid line)
pit.plot(time, CB, 'r-', label='Actual CB', markersize=5)       # Actual CB in red (solid line)
pit.plot(time, CC, 'c-', label='Actual CC', markersize=5)       # Actual CC in cyan (solid line)
pit.plot(time, CA_opt, 'm:', label='Modeled CA', linewidth=2)   # Modeled CA in magenta (dotted line)
pit.plot(time, CB_opt, 'b:', label='Modeled CB', linewidth=2)   # Modeled CB in blue (dotted line)
pit.plot(time, CC_opt, 'k:', label='Modeled CC', linewidth=2)   # Modeled CC in black (dotted line)

# Annotating k1 and k2 on the plot
pit.text(0.05, 0.95, f'Optimal k1: {k1_opt:.4f}', transform=pit.gca().transAxes, fontsize=12, verticalalignment='top')
pit.text(0.05, 0.90, f'Optimal k2: {k2_opt:.4f}', transform=pit.gca().transAxes, fontsize=12, verticalalignment='top')

pit.xlabel('Time')
pit.ylabel('Concentration')
pit.title('Optimized Parameters')
pit.legend()
pit.grid()
pit.show()
