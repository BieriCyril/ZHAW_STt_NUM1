import numpy as np
import matplotlib.pyplot as plt

# 1. Define the x range from -2 to 2
# We use 400 points to ensure the curve is smooth
x = np.linspace(-2, 2, 400)

# 2. Define the sigma values
sigmas = [1/4, 1/3, 1/2]

# 3. Create the plot
plt.figure(figsize=(10, 6))

for sigma in sigmas:
    # Calculate f(x) = e^(-x^2 / sigma)
    y = np.exp(-(x**2) / sigma)
    
    # Plot with a label for the legend
    plt.plot(x, y, label=f'σ = {sigma:.2f}')

# 4. Add axis labels, title, and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'Graph der Funktion $f(x) = e^{-x^2 / \sigma}$')
plt.legend()
plt.grid(True)

# 5. Show the plot
plt.show()