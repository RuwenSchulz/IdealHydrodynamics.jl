using DifferentialEquations, NLsolve, Plots, LeastSquaresOptim, LsqFit,Statistics, Optim, Interpolations, LinearAlgebra, BenchmarkTools, StaticArrays, Printf


# EoS
function PEoS(e,gamma)
    return (gamma - 1) * e
end

#inverse EoS
global function eEoS(P,gamma)
    return P/(gamma-1)
end 

function define_conserved(n, P, v,gamma)
    e = eEoS(P, gamma)
    n = n
    Q = v .* (P + e) 
    return e, n, Q
end

#analytic inversion
function primitive_from_conserved(e, n, Q,gamma)  
    n = n
    v = @. Q / max(gamma * e, 1e-12)
    P = PEoS(e,gamma)
    return n, P , v
end

#numerical inversion
#function primitive_from_conserved(e, n, Q)
#    len_e = length(e)
#    P_guess = ones(len_e)
#    v_guess = zeros(len_e)
#
#    conserved1(P) = eEoS(P)
#    conserved2(P,v)  = v .* (P .+ e)
#
#    function equations(vars)
#        P = vars[1:len_e]
#        v = vars[len_e+1:2len_e]
#        return vcat(
#            e .- conserved1(P),
#            Q .- conserved2(P,v)
#        )
#    end
#
#    sol = optimize(equations, vcat(P_guess, v_guess), LevenbergMarquardt())
#
#    P = sol.minimizer[1:len_e]
#    v = sol.minimizer[len_e+1:2*len_e]
#
#    P =  length(P) == 1 ? P[1] : P
#    v =  length(v) == 1 ? v[1] : v
#
#    return n, P, v
#end

function initial_conditions(Nx,gamma)
    n = ones(Nx)
    P = ones(Nx)
    v = zeros(Nx) .+0.0001

    # Left and right states
    n[1:Nx÷2] .= 1.0
    P[1:Nx÷2] .= 1.0

    n[Nx÷2+1:end] .= 0.001
    P[Nx÷2+1:end] .= 0.001

    # Compute conserved variables
    e, n, Q = define_conserved(n, P,v,gamma)

    return e, n, Q
end

# Flux function
function flux(e,n,Q,gamma)
    n, P, v = primitive_from_conserved(e, n, Q,gamma)
    f1 = v .* e .+ P .* v
    f2 = n .*v
    f3 = P .* (1 .+ v.^2)  .+ e .* v.^2
    return f1, f2, f3
end

function hll_flux(eL, nL, QL, eR, nR, QR,gamma)
    nL, PL, vL = primitive_from_conserved(eL, nL, QL,gamma)
    nR, PR, vR = primitive_from_conserved(eR, nR, QR,gamma)

    # Compute wave speeds
    cL = sqrt(gamma * PL / nL)
    cR = sqrt(gamma * PR / nR)
    
    SL = min(vL - cL, vR - cR)  # Left wave speed
    SR = max(vL + cL, vR + cR)  # Right wave speed
    SR = max(SR, SL + 1e-10)

    # Compute fluxes
    FL_e, FL_n, FL_Q = flux(eL, nL, QL,gamma)
    FR_e, FR_n, FR_Q = flux(eR, nR, QR,gamma)

    if SL > 0
        return FL_e, FL_n, FL_Q
    elseif SR < 0
        return FR_e, FR_n, FR_Q
    else
        F_e = (SR * FL_e - SL * FR_e + SL * SR * (eR - eL)) /  max(SR - SL, 1e-10)
        F_n = (SR * FL_n - SL * FR_n + SL * SR * (nR - nL)) /  max(SR - SL, 1e-10)
        F_Q = (SR * FL_Q - SL * FR_Q + SL * SR * (QR - QL)) /  max(SR - SL, 1e-10)
        return F_e, F_n, F_Q
    end
end

function rusanov_flux(eL, nL, QL, eR, nR, QR,gamma)
    nL, PL, vL = primitive_from_conserved(eL, nL, QL,gamma)
    nR, PR, vR = primitive_from_conserved(eR, nR, QR,gamma)


    # Compute sound speeds
    cL = sqrt(gamma * PL / nL)
    cR = sqrt(gamma * PR / nR)

    # Max wave speed
    Smax = max(abs(vL) + cL, abs(vR) + cR)

    # Compute fluxes
    FL_e, FL_n, FL_Q = flux(eL, nL, QL,gamma)
    FR_e, FR_n, FR_Q = flux(eR, nR, QR,gamma)

    # Rusanov (Local Lax-Friedrichs) flux
    F_e = 0.5 * (FL_e + FR_e) - 0.5 * Smax * (eR - eL)
    F_n = 0.5 * (FL_n + FR_n) - 0.5 * Smax * (nR - nL)
    F_Q = 0.5 * (FL_Q + FR_Q) - 0.5 * Smax * (QR - QL)

    return F_e, F_n, F_Q
end

function first_order_euler!(flux_updater,e, n, Q, dx, dt, gamma)
    Nx = length(e)

    # Flux storage
    F_e = zeros(Nx-1)
    F_n = zeros(Nx-1)
    F_Q = zeros(Nx-1)
    # Compute fluxes at cell interfaces
    for i in 1:Nx-1
        F_e[i], F_n[i], F_Q[i] = flux_updater(e[i], n[i], Q[i],
                                              e[i+1], n[i+1], Q[i+1], gamma)
    end

    # Update conserved variables
    for i in 2:Nx-1
        e[i] -= dt / dx * (F_e[i] - F_e[i-1])
        n[i] -= dt / dx * (F_n[i] - F_n[i-1])
        Q[i] -= dt / dx * (F_Q[i] - F_Q[i-1])
    end

    return e, n, Q
end

function hydro_rhs!(du, u, p, t)
    Nx = length(u) ÷ 3 
    dx, gamma, flux_updater = p  

    @show t
    # Flux storage
    F1 = zeros(Nx-1)
    F2 = zeros(Nx-1)
    F3 = zeros(Nx-1)

    e = @view u[1:Nx]
    n = @view u[Nx+1:2Nx]
    Q = @view u[2Nx+1:3Nx]

    de = @view du[1:Nx]
    dn = @view du[Nx+1:2Nx]
    dQ = @view du[2Nx+1:3Nx]

    @inbounds @simd for i in 1:Nx-1
        Q1_L, Q2_L, Q3_L = e[i], n[i], Q[i]
        Q1_R, Q2_R, Q3_R = e[i+1], n[i+1], Q[i+1]
        F1[i], F2[i], F3[i] = flux_updater(Q1_L, Q2_L, Q3_L, Q1_R, Q2_R, Q3_R,gamma)
    end

    @inbounds @simd for i in 2:Nx-1
        de[i] = -1 / dx * (F1[i] - F1[i-1])
        dn[i] = -1 / dx * (F2[i] - F2[i-1])
        dQ[i] = -1 / dx * (F3[i] - F3[i-1])
    end


end

function solve_system(Nx, flux_updater;time_updater =:tsit5)
    L = 1.0
    dx = L / Nx
    Tfinal = 0.2
    gamma = 5/3

    # Initialize from meaningful (n, v, P)
    e, n, Q = initial_conditions(Nx,gamma)

    if time_updater == :tsit5
        u0 = vcat(e, n, Q)  # Store all variables in a single vector
        tspan = (0.0, Tfinal)
        p = (dx, gamma, flux_updater)
        prob = ODEProblem(hydro_rhs!, u0, tspan, p)
        sol = solve(prob, Tsit5(); reltol=1e-12, abstol=1e-12)

        u1 = []
        u2 = []
        u3 = []
        num_steps = length(sol.t)

        for i in 1:num_steps
            u = sol.u[i]
            ee = u[1:Nx]
            n = u[Nx+1:2Nx]
            Q = u[2Nx+1:3Nx]
            u1_i, u2_i, u3_i =  primitive_from_conserved(ee, n, Q,gamma)
            push!(u1, u1_i)
            push!(u2, u2_i)
            push!(u3, u3_i)
        end

        return [sol.t,u1,u2,u3]
    end

    # Time-stepping loop for SSP-RK3
    t = 0.0
    u1 = []
    u2 = []
    u3 = []
    times = []
    dt = 0.001

    while t < Tfinal
        e,n,Q  = time_updater(flux_updater, e, n, Q, dx, dt, gamma)

        u1_i,u2_i,u3_i = primitive_from_conserved(e, n, Q,gamma)
        push!(u1, copy(u1_i))
        push!(u2, copy(u2_i))
        push!(u3, copy(u3_i))
        push!(times, t)
        t += dt
    end
    return [times, u1, u2, u3]
end

Nx=50

result_tsit5_rusanov = solve_system(Nx, rusanov_flux;time_updater = first_order_euler!)
result_tsit5_hll     = solve_system(Nx, hll_flux,time_updater = first_order_euler!)

result_1Euler_rusanov = solve_system(Nx, rusanov_flux)
result_1Euler_hll     = solve_system(Nx, hll_flux)


function match_time_series(target_times, source_data)
    t_src, n_src, P_src, v_src = source_data
    matched_indices = [findmin(abs.(t_src .- t))[2] for t in target_times]
    t_matched = t_src[matched_indices]
    n_matched = n_src[matched_indices]
    P_matched = P_src[matched_indices]
    v_matched = v_src[matched_indices]
    return (t_matched, n_matched, P_matched, v_matched)
end


result_tsit5_hll_matched = match_time_series(result_tsit5_rusanov[1], result_tsit5_hll)
result_1Euler_rusanov_matched = match_time_series(result_tsit5_rusanov[1], result_1Euler_rusanov)
result_1Euler_hll_matched = match_time_series(result_tsit5_rusanov[1], result_1Euler_hll)

result_tsit5_rusanov_relativistic_matched = match_time_series(result_tsit5_rusanov[1], result_tsit5_rusanov_relativistic)
result_1Euler_rusanov_relativistic_matched = match_time_series(result_tsit5_rusanov[1], result_1Euler_rusanov_relativistic)

result_vector = [result_tsit5_rusanov,result_tsit5_hll_matched,result_1Euler_rusanov_matched,result_1Euler_hll_matched,result_tsit5_rusanov_relativistic_matched,result_1Euler_rusanov_relativistic_matched]
string_vector = ["result_tsit5_rusanov","result_tsit5_hll_matched","result_1Euler_rusanov_matched","result_1Euler_hll_matched","result_tsit5_rusanov_relativistic_matched","result_1Euler_rusanov_relativistic_matched"]


function plot_results(results_sets, labels::Vector{String}; frames=50, filename="animation.gif")
    @assert length(results_sets) == length(labels) "Each results set must have a corresponding label."

    sampled_results = [[res_i[1:Int(floor(length(res_i)/frames)):end] for res_i in res] for res in results_sets]

    colors = [:red, :blue, :purple, :orange,:green,:black]

    num_frames = length(sampled_results[1][1])  # Assume all have same number of time steps
    anim = @animate for i in 1:num_frames
        plt_n = plot(title="Density (n)" , xlabel="x", ylabel="n", legend=:bottomleft,ylims=(0, 1))
        plt_v = plot(title="Velocity (v)", xlabel="x", ylabel="v", legend=:bottomleft,ylims=(0, 1))
        plt_P = plot(title="Pressure (P)", xlabel="x", ylabel="P", legend=:bottomleft,ylims=(0, 1))

        for (results, label, color) in zip(sampled_results, labels, colors)
            t, nn, P, v = results
            plot!(plt_n, nn[i], label=label, color=color, lw=2)
            plot!(plt_v, v[i], label=label, color=color, lw=2)
            plot!(plt_P, P[i], label=label, color=color, lw=2)
        end

        plot(plt_n, plt_v, plt_P, layout=(1, 3), size=(1200, 400))
    end

    gif(anim, filename, fps=5)
end

plot_results(result_vector,string_vector;frames=40)

