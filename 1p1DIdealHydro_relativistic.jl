using DifferentialEquations, NLsolve, Plots, LeastSquaresOptim, LsqFit,
      Statistics, Optim, Interpolations, LinearAlgebra, BenchmarkTools,
      StaticArrays, Printf

# -------------------------
# Problem Setup & Constants
# -------------------------

gamma = 5/3  # Adiabatic index for ideal gas

# Lorentz factor: γ = 1 / sqrt(1 - v^2)
@inline function γ(v)
    return 1.0 ./ sqrt.(1 .- (v .^ 2))
end

# Ideal gas equation of state (pressure as function of energy density)
@inline function P(e)
    return (gamma - 1) * e
end

# Inverse of equation of state: energy density from pressure
@inline global function e(P)
    return P / (gamma - 1)
end

# ------------------------------
# Energy-Momentum and Currents
# ------------------------------

# Tμν components and current densities (relativistic expressions)

@inline function T00_fun!(result, n, P, v)
    result .= -P .+ (P .+ e(P)) .* γ(v).^2
end

@inline function T10_fun!(result, n, P, v)
    result .= v .* (P .+ e(P)) .* γ(v).^2
end

@inline function T01_fun!(result, n, P, v)
    result .= v .* (P .+ e(P)) .* γ(v).^2
end

@inline function T11_fun!(result, n, P, v)
    result .= P .+ v.^2 .* (P .+ e(P)) .* γ(v).^2
end

@inline function J0_fun!(result, n, P, v)
    result .= n .* γ(v)
end

@inline function J1_fun!(result, n, P, v)
    result .= n .* v .* γ(v)
end

# Non-mutating versions (return new arrays)
@inline T00_fun(n, P, v) = -P .+ (P .+ e(P)) .* γ(v).^2
@inline T10_fun(n, P, v) = v .* (P .+ e(P)) .* γ(v).^2
@inline T01_fun(n, P, v) = v .* (P .+ e(P)) .* γ(v).^2
@inline T11_fun(n, P, v) = P .+ v.^2 .* (P .+ e(P)) .* γ(v).^2
@inline J0_fun(n, P, v) = n .* γ(v)
@inline J1_fun(n, P, v) = n .* v .* γ(v)

# ---------------------
# Structs for Clarity
# ---------------------

struct HydroState
    n::Vector{Float64}
    P::Vector{Float64}
    v::Vector{Float64}
end

struct ConservedVars
    Q1::Vector{Float64}  # Energy density (T00)
    Q2::Vector{Float64}  # Momentum density (T01)
    Q3::Vector{Float64}  # Charge density (J0)
end

struct FluxVars
    F1::Vector{Float64}  # Momentum flux (T10)
    F2::Vector{Float64}  # Stress (T11)
    F3::Vector{Float64}  # Charge flux (J1)
end


# ---------------------------
# Flux & State Conversion
# ---------------------------

@inline fluxes_from_state(n, P, v) = (T10_fun(n, P, v), T11_fun(n, P, v), J1_fun(n, P, v))

@inline function fluxes_from_conserved(Q1, Q2, Q3)
    n, P, v = primitive_from_conserved(Q1, Q2, Q3)
    return fluxes_from_state(n, P, v)
end

@inline function conserved_from_state(state::HydroState)
    return (T00_fun(state.n, state.P, state.v),
            T01_fun(state.n, state.P, state.v),
            J0_fun(state.n, state.P, state.v))
end

# ----------------------------------
# Primitive Variable Inversion (NR)
# ----------------------------------

@inline function primitive_from_conserved(Q1,Q2,Q3)
    local_Nx = length(Q1)
    
    n_guess = ones(local_Nx)
    P_guess = ones(local_Nx)
    v_guess = fill(0.001,local_Nx)

    T00_values  = similar(n_guess)
    T01_values  = similar(n_guess)
    J0_values   = similar(n_guess)

    idx1, idx2, idx3 = 1:local_Nx, (local_Nx+1):(2*local_Nx), (2*local_Nx+1):(3*local_Nx)

    function equations(vars)

        n,P,v = vars[idx1], vars[idx2], vars[idx3]

        T00_fun!(T00_values,n,P,v)
        T01_fun!(T01_values,n,P,v)
        J0_fun!(J0_values,n,P,v)

        return vcat(
            Q1 .- T00_values,
            Q2 .- T01_values,
            Q3 .- J0_values
        )
    end

    sol = optimize(equations, vcat(n_guess, P_guess, v_guess), LevenbergMarquardt())
    n = sol.minimizer[1:local_Nx]
    P = sol.minimizer[local_Nx+1:2local_Nx]
    v = sol.minimizer[2local_Nx+1:3local_Nx]

    n =  length(n) == 1 ? n[1] : n
    P =  length(P) == 1 ? P[1] : P
    v =  length(v) == 1 ? v[1] : v


    return n,P,v
end

# -------------------
# Rusanov Flux (1D)
# -------------------

function rusanov_flux(Q1_L, Q2_L, Q3_L, Q1_R, Q2_R, Q3_R)
    nL, PL, vL = primitive_from_conserved(Q1_L, Q2_L, Q3_L)
    nR, PR, vR = primitive_from_conserved(Q1_R, Q2_R, Q3_R)

    csL= sqrt(sound_speed_squared(e(PL),PL))
    csR= sqrt(sound_speed_squared(e(PR),PR))

    # Compute relativistic wave speeds
    λ_plus_L, λ_minus_L = relativistic_wave_speeds(vL, csL)
    λ_plus_R, λ_minus_R = relativistic_wave_speeds(vR, csR)

    # Max signal speed for Rusanov dissipation
    Smax = max(abs(λ_plus_L), abs(λ_minus_L), abs(λ_plus_R), abs(λ_minus_R))

    # Compute fluxes
    F1_L, F2_L, F3_L = fluxes_from_state(nL, PL, vL)
    F1_R, F2_R, F3_R = fluxes_from_state(nR, PR, vR)

    # Rusanov (Local Lax-Friedrichs) flux
    F1 = 0.5 * (F1_L + F1_R) - 0.5 * Smax * (Q1_R - Q1_L)
    F2 = 0.5 * (F2_L + F2_R) - 0.5 * Smax * (Q2_R - Q2_L)
    F3 = 0.5 * (F3_L + F3_R) - 0.5 * Smax * (Q3_R - Q3_L)

    return F1, F2, F3
end

# ----------------------
# Initial Conditions
# ----------------------

function initialize_state(Nx)
    n = fill(1.0, Nx)
    P = fill(1.0, Nx)
    v = fill(1e-10, Nx)

    n[Nx÷2+1:end] .= 0.125
    P[Nx÷2+1:end] .= 0.1

    return HydroState(n, P, v)
end

# -------------------------
# Time Integration (Tsit5)
# -------------------------

function hydro_rhs!(du, u, p, t)
    dx, Nx = p

    F1, F2, F3 = zeros(Nx-1), zeros(Nx-1), zeros(Nx-1)

    Q1 = u[1:Nx]
    Q2 = u[Nx+1:2Nx]
    Q3 = u[2Nx+1:3Nx]

    for i in 1:Nx-1
        F1[i], F2[i], F3[i] = rusanov_flux(Q1[i], Q2[i], Q3[i], Q1[i+1], Q2[i+1], Q3[i+1])
    end

    @inbounds @simd for i in 2:Nx-1
        du[i]        = -1 / dx * (F1[i] - F1[i-1])
        du[Nx+i]     = -1 / dx * (F2[i] - F2[i-1])
        du[2Nx + i]  = -1 / dx * (F3[i] - F3[i-1])
    end

    return nothing
end

function solve_system_Tsit5(Nx)
    L = 1.0
    dx = L / Nx
    t_final = 0.2

    init = initialize_state(Nx)
    Q1, Q2, Q3 = conserved_from_state(init)

    u0 = vcat(Q1, Q2, Q3)
    prob = ODEProblem(hydro_rhs!, u0, (0.0, t_final), (dx, Nx))
    sol = solve(prob, Tsit5(); reltol=1e-12, abstol=1e-12)

    u1, u2, u3 = [], [], []

    for u in sol.u
        n, P, v = primitive_from_conserved(u[1:Nx], u[Nx+1:2Nx], u[2Nx+1:3Nx])
        push!(u1, n); push!(u2, P); push!(u3, v)
    end

    return [sol.t, u1, u2, u3]
end


# --------------------------
# First-Order Euler Method
# --------------------------

function sound_speed_squared(e, Pval)
    δe = 1e-6 .* e
    dPde = (P.(e .+ δe) .- Pval) ./ δe
    return dPde
end

function relativistic_wave_speeds(v, cs)
    λ_plus = (v + cs) / (1 + v * cs)
    λ_minus = (v - cs) / (1 - v * cs)
    return λ_plus, λ_minus
end

function compute_dt(Q1, Q2, Q3, dx, CFL)
    n, P, v = primitive_from_conserved(Q1, Q2, Q3)
    cs = sqrt.(sound_speed_squared.(e(P), P))
    λ_plus, λ_minus = relativistic_wave_speeds.(v, cs)
    Smax = maximum(max.(abs.(λ_plus), abs.(λ_minus)))
    return CFL * dx / Smax
end

function first_order_euler_time_step(Q1,Q2,Q3, dx, dt)
    Nx = length(Q1)

    # Flux storage
    F1 = zeros(Nx-1)
    F2 = zeros(Nx-1)
    F3 = zeros(Nx-1)

    @inbounds @simd for i in 1:Nx-1
        Q1_L, Q2_L, Q3_L = Q1[i], Q2[i], Q3[i]
        Q1_R, Q2_R, Q3_R = Q1[i+1], Q2[i+1], Q3[i+1]
        F1[i], F2[i], F3[i] = rusanov_flux(Q1_L, Q2_L, Q3_L, Q1_R, Q2_R, Q3_R)
    end

    @inbounds @simd for i in 2:Nx-1
        Q1[i] -= dt / dx * (F1[i] - F1[i-1])
        Q2[i] -= dt / dx * (F2[i] - F2[i-1])
        Q3[i] -= dt / dx * (F3[i] - F3[i-1])
    end

    return Q1,Q2,Q3
end

function solve_system_euler(Nx)
    L = 1.0
    dx = L / Nx
    t_final = 0.2
    initial_state =  initialize_state(Nx)
    Q1,Q2,Q3 = conserved_from_state(initial_state)
    result = [Float64[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]]

    CFL = 0.2
    dt = 0.001
    t = 0.0
    while t < t_final

        Q1,Q2,Q3 =  first_order_euler_time_step(Q1,Q2,Q3, dx, dt)
        push!(result[1], t)
        push!(result[2],primitive_from_conserved(Q1,Q2,Q3)[1])
        push!(result[3],primitive_from_conserved(Q1,Q2,Q3)[2])
        push!(result[4],primitive_from_conserved(Q1,Q2,Q3)[3])
        dt = compute_dt(Q1,Q2,Q3,dx,CFL)
        t += dt
        #println(t)
    end
    return result
end


# ------------------------
# Run Simulations
# ------------------------

Nx = 60

# Solve using Tsit5 adaptive RK
result_tsit5_rusanov_relativistic = solve_system_Tsit5(Nx)

# Solve using first-order Euler with CFL
result_1Euler_rusanov_relativistic = solve_system_euler(Nx)

# Result now holds time and (n, P, v) histories
result_tsit5_rusanov_relativistic
result_1Euler_rusanov_relativistic

# ------------------------
# Plotting Results
# ------------------------

include("plotting.jl")
result_1Euler_rusanov_relativistic_matched = match_time_series(result_tsit5_rusanov_relativistic[1], result_1Euler_rusanov_relativistic)
result_1Euler_rusanov_relativistic_matched

result_vector = [result_tsit5_rusanov_relativistic, result_1Euler_rusanov_relativistic_matched]
string_vector = ["Tsit5 Rusanov Relativistic", "1st Order Euler Rusanov Relativistic"]
plot_results(result_vector, string_vector; frames=40)