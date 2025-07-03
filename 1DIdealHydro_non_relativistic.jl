using DifferentialEquations
using Plots

# Define parameters
γ = 4/3  # Adiabatic index

# System of PDEs
function hydro_system!(du, u, p, t)
    ε, v, n = u  # Unpack state variables
    
    P = (γ - 1) * ε  # EoS
    # Fluxes
    F1 = (ε + P) * v
    F2 = (ε + P) * v^2 + P
    F3 = n * v

    # Conservation laws in 1D
    du[1] = -F1
    du[2] = -F2
    du[3] = -F3
end

# Custom Euler solver
function euler_solve(f!, u0, tspan, dt)
    t = tspan[1]:dt:tspan[2]
    n = length(t)
    m = length(u0)
    u = zeros(m, n)
    u[:,1] = u0
    
    du = similar(u0)
    
    for i in 1:(n-1)
        f!(du, u[:,i], nothing, t[i])
        u[:,i+1] = u[:,i] + dt * du
    end
    
    return t, u
end


# Solve using DifferentialEquations.jl
tspan = (0.0, 1.5)
u0 = [1.0, 0.1, 1.0]  # Initial condition at x=0.5
prob = ODEProblem(hydro_system!, u0, tspan)

sol_tsit5 = solve(prob, Tsit5())

dt = 0.001  # Small time step for stability
t_euler, u_euler = euler_solve(hydro_system!, u0, tspan, dt)

p1 = plot(title="Energy Density Comparison")
plot!(p1, sol_tsit5, vars=(0,1), label="Tsit5", linewidth=2)
plot!(p1, t_euler, u_euler[1,:], label="Euler", linestyle=:dash)
xlabel!(p1, "Time")
ylabel!(p1, "Energy Density")

p2 = plot(title="Velocity Comparison")
plot!(p2, sol_tsit5, vars=(0,2), label="Tsit5", linewidth=2)
plot!(p2, t_euler, u_euler[2,:], label="Euler", linestyle=:dash)
xlabel!(p2, "Time")
ylabel!(p2, "Velocity")

p3 = plot(title="Number Density Comparison")
plot!(p3, sol_tsit5, vars=(0,3), label="Tsit5", linewidth=2)
plot!(p3, t_euler, u_euler[3,:], label="Euler", linestyle=:dash)
xlabel!(p3, "Time")
ylabel!(p3, "Number Density")

# Combine plots
plot(p1, p2, p3, layout=(3,1), size=(800,1000))

