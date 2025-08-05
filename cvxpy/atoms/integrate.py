"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import numpy as np

import cvxpy as cvx


def numerical_integration_1d(function, a, b, n=1000, method="trapezoid"):
    """Numerically integrate f(x) from a to b using the specified method.
    
    Args:
        function: Callable f(x) returning a scalar or CVXPY expression.
        a, b: Integration limits (b > a).
        n: Number of subintervals (must be even for Simpson's rule).
        method: One of ["left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons"].
    
    Returns:
        CVXPY expression or numeric value representing the integral.
    """
    # === Input Validation ===
    if a == b:
        return cvx.Constant(0)
    
    if n <= 0:
        raise ValueError("n must be positive.")
    if b < a:
        raise ValueError(f"b ({b}) must be ≥ a ({a}).")
    if method not in ["left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons"]:
        raise ValueError(f"Unsupported method: {method}")

    h = (b - a) / n  # Subinterval width

    # === Convert Numeric Values to CVXPY Constants ===
    def to_expr(val):
        return cvx.Constant(val) if isinstance(val, (int, float, np.number)) else val

    # === Integration Methods ===
    if method == "left_riemann":
        x = np.linspace(a, b - h, n)  # Left endpoints
    elif method == "right_riemann":
        x = np.linspace(a + h, b, n)  # Right endpoints
    elif method == "midpoint":
        x = np.linspace(a + h/2, b - h/2, n)  # Midpoints
    elif method == "trapezoid":
        x = np.linspace(a, b, n + 1)  # All endpoints
    elif method == "simpsons":
        if n % 2 != 0:
            n += 1  # Force even n for Simpson's
            h = (b - a) / n
        x = np.linspace(a, b, n + 1)  # All endpoints

    # Evaluate f(x) at all sample points
    y = [to_expr(function(xi)) for xi in x]

    # Compute the integral approximation
    if method in ["left_riemann", "right_riemann", "midpoint"]:
        return h * cvx.sum(y)
    elif method == "trapezoid":
        s = cvx.sum(y[1:-1]) if n > 1 else cvx.Constant(0)
        return h * (0.5 * (y[0] + y[-1]) + s)
    elif method == "simpsons":
        odd_sum = cvx.sum(y[1:-1:2])
        even_sum = cvx.sum(y[2:-2:2]) if n > 2 else cvx.Constant(0)
        return (h / 3) * (y[0] + y[-1] + 4 * odd_sum + 2 * even_sum)