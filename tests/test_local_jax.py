# test_local_jax.py
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX location: {jax.__file__}")

# Simple computation test
x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
print(f"Array addition: {x + y}")