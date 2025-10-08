import sys
import numpy as np
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")

import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX location: {jax.__file__}")

# Simple computation test
x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
print(f"Array addition: {x + y}")

"""
Python: 3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]
NumPy: 2.3.3
JAX version: 0.8.0.dev20251007+45e1fa079
JAX location: C:\Users\Sheil\Development\jax\jax\__init__.py
Array addition: [5 7 9]
(jax-dev) PS C:\Users\Sheil\Development\jax> 
"""