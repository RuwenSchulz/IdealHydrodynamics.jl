# IdealHydrodynamics.jl

A Julia package for solving 1D ideal hydrodynamics equations using finite volume methods with support for Rusanov and HLL flux solvers.

---

##  Features

- First-order Euler and Tsit5 (adaptive Runge-Kutta) time integration
- Support for Rusanov and HLL Riemann solvers
- Primitive/conserved variable transformations
- Energy-conserving formulation with ideal equation of state
- Built-in plotting and animation utilities

---

##  Installation

```julia
using Pkg
Pkg.add(url="https://github.com/RuwenSchulz/IdealHydrodynamics.jl")
```

---
## Example Output

Here is an example result of the Sod shock tube simulation:

![Shock tube simulation](https://github.com/RuwenSchulz/IdealHydrodynamics.jl/preview/sod_shock.png)

---
