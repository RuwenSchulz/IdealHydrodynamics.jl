using Plots, Printf

# Match time series between datasets by nearest timestamps
# This is useful when comparing methods with different time grids (e.g., adaptive vs. fixed time stepping)
function match_time_series(target_times, source_data)
    t_src, n_src, P_src, v_src = source_data

    # For each target time, find the closest index in the source time series
    matched_indices = [findmin(abs.(t_src .- t))[2] for t in target_times]

    # Extract the matched values from the source data
    t_matched = t_src[matched_indices]
    n_matched = n_src[matched_indices]
    P_matched = P_src[matched_indices]
    v_matched = v_src[matched_indices]

    return (t_matched, n_matched, P_matched, v_matched)
end

# Plot the evolution of density, velocity, and pressure over time
# across multiple result sets. Saves animation as a GIF.
function plot_results(results_sets, labels::Vector{String}; frames=50, filename="animation.gif")
    @assert length(results_sets) == length(labels)  # Ensure labels match result sets

    # Downsample each dataset to fit desired number of frames
    sampled_results = [[res_i[1:Int(floor(length(res_i)/frames)):end] for res_i in res]
                       for res in results_sets]

    # Define distinct colors for each method being plotted
    colors = [:red, :blue, :purple, :orange, :green, :black]

    # Number of frames to animate (assumed equal across datasets)
    num_frames = length(sampled_results[1][1])

    # Build animation
    anim = @animate for i in 1:num_frames
        # Create individual subplots for each quantity
        plt_n = plot(title="Density (n)", xlabel="x", ylabel="n", legend=:bottomleft, ylims=(0, 1))
        plt_v = plot(title="Velocity (v)", xlabel="x", ylabel="v", legend=:bottomleft, ylims=(0, 1))
        plt_P = plot(title="Pressure (P)", xlabel="x", ylabel="P", legend=:bottomleft, ylims=(0, 1))

        # Loop through each result set and plot at current frame `i`
        for (results, label, color) in zip(sampled_results, labels, colors)
            t, nn, P, v = results
            plot!(plt_n, nn[i], label=label, color=color, lw=2)
            plot!(plt_v, v[i], label=label, color=color, lw=2)
            plot!(plt_P, P[i], label=label, color=color, lw=2)
        end

        # Combine all three subplots into one frame
        plot(plt_n, plt_v, plt_P, layout=(1, 3), size=(1200, 400))
    end

    # Save the animation as a GIF file
    gif(anim, filename, fps=5)
end

# Example usage to generate and save the animation
#plot_results(result_vector, string_vector; frames=40)
