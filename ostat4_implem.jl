using Random, Distributions, Statistics, Plots, StatsBase, KernelDensity, HypothesisTests


function create_sequence_and_subsequences(k::Int, weights::Vector{Int})
    # Step 1: Create the main sequence
    sequence = collect(1:k)  # Sequence from 1 to k

    # Step 2: Create subsequences based on weights
    subsequences =  Vector{Vector{Int}}()
    current_index = 1
    for weight in weights
        end_index = current_index + weight - 1
        if end_index > k
            error("Weights exceed the length of the sequence!")
        end
        push!(subsequences, collect(current_index:end_index))
        current_index = end_index + 1
    end

    # Check if all elements of the sequence are covered
    if current_index - 1 != k
        error("Weights do not perfectly partition the sequence!")
    end

    return sequence, subsequences
end

function generate_rv(selected_distribution::String)
    # Step 5: Generate a random variable (RV) based on the selected distribution
    if selected_distribution == "Normal"
        stddev = rand(1.0:0.5:10.0)
        mean = rand(3*stddev:35.0)
        rv = rand(Normal(mean, stddev))  # Generate RV from Normal distribution
    elseif selected_distribution == "LogNormal"
        log_mean = rand(1.0:0.5:5.0)
        log_stddev = rand(0.1:0.1:1.0)
        rv = rand(LogNormal(log_mean, log_stddev))  # Generate RV from LogNormal 
        # distribution
    elseif selected_distribution == "Exponential"
        theta = rand(0.2:0.01:1.0)
        rv = rand(Exponential(theta))  # Generate RV from Exponential distribution
    elseif selected_distribution == "Pareto"
        alpha = rand(2.3:0.5:5.3)  # Shape parameter
        theta = rand(2.1:0.5:4.1)  # Scale parameter
        rv = rand(Pareto(alpha, theta))  # Generate RV from Pareto distribution
    elseif selected_distribution == "Weibull"
        shape = rand(2.0:0.5:5.0)
        scale = rand(1.0:0.5:6.0)
        rv = rand(Weibull(shape, scale))  # Generate RV from Weibull distribution
    end
    return rv
end


# Function to delete an element and handle logic
function delete_and_generate_rv(element::Int64, sequence::Vector{Int64},
    subsequences::Vector{Vector{Int64}}, distributions::Vector{String})
    # Check if the element exists in the sequence
    if !(element in sequence)
        println("Element $element not found in the sequence!")
        return nothing
    end

    # Remove the element from the sequence
    deleteat!(sequence, findfirst(x -> x == element, sequence))

    # Find which subsequence contains the element
    for (i, subseq) in enumerate(subsequences)
        if element in subseq
            # Remove the element from the subsequence
            deleteat!(subseq, findfirst(x -> x == element, subseq))

            # Generate an RV using the corresponding distribution
            rv = generate_rv(distributions[i])

            return rv
        end
    end
end


function generalized_random_selection(k::Int64, weights::Vector{Int64}, 
    distributions::Vector{String})
    
    sequence, subsequences=create_sequence_and_subsequences(k::Int, 
    weights::Vector{Int})
    
    # Main loop to select numbers from the sequence until all 
    # numbers are exhausted
    selected_rvs = Float64[]  # To store generated random variables
    
    while !isempty(sequence)

        element=rand(sequence)
        # Select a random number and corresponding distribution
        rv= delete_and_generate_rv(element, sequence, subsequences, distributions)
        
        # Append the generated random variable to the list
        push!(selected_rvs, rv)
        
    end
    
    return selected_rvs
end

# Visualization: Empirical CDF and KDE
function plot_kde(data)
    
    kde_result = kde(data)
    x_kde = kde_result.x
    y_kde = kde_result.density
    plot(x_kde, y_kde, label="KDE", xlabel="Claim Size", 
    ylabel="Density", color=:blue)
    savefig("kde_plot_2.png")
end

# Log-likelihood calculation for distribution fitting
function compute_log_likelihood(distribution, data)
    return sum(logpdf(distribution, data))
end

# AIC calculation based on log-likelihood and number of parameters
function compute_aic(log_likelihood, n_params, n_data_points)
    return 2 * n_params - 2 * log_likelihood
end


# Kolmogorov-Smirnov test to compare empirical data with a distribution
function ks_test_distribution(distribution, data)
    # Using the StatsTests package
    p_val=ApproximateOneSampleKSTest(data,distribution)
    return p_val
end


function extract_parameters(dist)
    if dist isa Normal
        # For Normal distribution, access the parameters directly
        return (mu=mean(dist), sigma=std(dist))
    elseif dist isa LogNormal
        # For LogNormal distribution, access the parameters directly
        return (mu=meanlogx(dist), sigma=stdlogx(dist))
    elseif dist isa Exponential
        # For Exponential distribution, use rate directly
        return (lambda=rate(dist))
    elseif dist isa Pareto
        # For Pareto distribution, access the parameters directly
        return (alpha=shape(dist), scale_=scale(dist))
    elseif dist isa Weibull
        # For Weibull distribution, access the parameters directly
        return (shape=shape(dist), scale=scale(dist))
    else
        error("Unsupported distribution type")
    end
end

using Plots

# Function to plot data with a given distribution fit (CDF and PDF)
function plot_with_distribution(data, dist, label, fig_title, filename)
    # Empirical CDF
    empirical_cdf = ecdf(data)
    x_values = sort(data)
    y_values = empirical_cdf.(x_values)  # Get corresponding cumulative probabilities
    
    # Create the plot
    p = plot(x_values, y_values, xlabel="Claim Size", ylabel="Cumulative Probability", 
    label="Empirical CDF", color=:black, linewidth=2)
    
    # Plot the fitted distribution CDF
    plot!(x_values, cdf(dist, x_values), label="Fitted CDF ($label)", 
    color=:red, linewidth=2)
    
    # Plot the fitted distribution PDF
    plot!(x_values, pdf(dist, x_values), label="Fitted PDF ($label)", 
    color=:blue, linewidth=2)
    
    # Set the title for the plot
    title!(fig_title)
    
    # Save the plot to a file
    savefig(filename)
    
    return p
end

# Function to generate 5 random distributions with varying parameters
function generate_candidate_distributions()
    candidate_distributions = []

    # Generate 5 random Normal distributions
    for _ in 1:5
        stddev_normal = rand(1.0:0.5:10.0)
        mean_normal = rand(1.0:3*stddev_normal:35.0)
        push!(candidate_distributions, Normal(mean_normal, stddev_normal))
    end

    # Generate 5 random LogNormal distributions
    for _ in 1:5
        log_mean_lognormal = rand(1.0:0.5:5.0)
        log_stddev_lognormal = rand(0.1:0.1:1.0)
        push!(candidate_distributions, LogNormal(log_mean_lognormal,
         log_stddev_lognormal))
    end

    # Generate 5 random Exponential distributions
    for _ in 1:5
        theta_exponential = rand(0.2:0.01:1.0)
        push!(candidate_distributions, Exponential(theta_exponential))
    end

    # Generate 5 random Pareto distributions
    for _ in 1:5
        alpha_pareto = rand(2.3:0.5:5.3)  # Shape parameter
        theta_pareto = rand(2.1:0.5:4.1)  # Scale parameter
        push!(candidate_distributions, Pareto(alpha_pareto, theta_pareto))
    end

    # Generate 5 random Weibull distributions
    for _ in 1:5
        shape_weibull = rand(2.0:0.5:5.0)
        scale_weibull = rand(1.0:0.5:6.0)
        push!(candidate_distributions, Weibull(shape_weibull, scale_weibull))
    end

    return candidate_distributions
end

using CSV

# Function to save the best KS test distribution and its 
# parameters to a file
function save_best_ks_distribution_as_string(dist, filename=
    "best_ks_distribution.txt")
    # Convert the distribution to a string (this includes its 
    # name and parameters)
    dist_str = string(dist)  # This will give something like 
    # "LogNormal{Float64}(μ=2.0, σ=0.9)"
    
    open(filename, "w") do file
        write(file, "$dist_str\n")
    end
    
    println("Best KS distribution saved to $filename")
end



Random.seed!(770)

# Example usage
k = 120  # Total number of random values to generate
n = 5  # Number of distributions

weights = rand(1:10,n) # Example weights


weights= (weights.*k)./sum(weights)

println("weights:",weights)
weights = floor.(weights)

left=k-sum(weights)

weights[rand(1:5)]+=left
weights= Int64.(round.(weights))

#the above part is to normalize the weights w.r.t k so that ,they
#  are integers in the end too

distributions = ["Normal", "LogNormal", "Exponential", "Pareto", "Weibull"]  
# Distributions chosen

# Call the generalized function
selected_rvs = generalized_random_selection(k, weights, distributions)


# Visualization of Claim Size Data
histogram(selected_rvs, bins=50, norm=true, alpha=0.7, label="Claim Size Data")


# Generate the 5 distributions for each type
candidate_distributions = generate_candidate_distributions()

aic_results = []
log_likelihood_results = []
ks_results = []

for dist in candidate_distributions
    log_likelihood = compute_log_likelihood(dist, selected_rvs)
    params = extract_parameters(dist)
    n_params = length(params)
    aic = compute_aic(log_likelihood, n_params, length(selected_rvs))
    ks_result = ks_test_distribution(dist, selected_rvs)
    p_value = pvalue(ks_result)
    println("P-value for $dist: ", p_value)
    push!(aic_results, aic)
    push!(log_likelihood_results, log_likelihood)
    push!(ks_results, ks_result)
end

# Function to randomly select n numbers from the list of claim sizes
function select_random_claims(claim_sizes::Vector{Float64}, n::Int)
    if n > length(claim_sizes)
        error("n cannot be larger than the total number of claim sizes available!")
    end
    return rand(claim_sizes, n)  # Randomly select n elements from the list
end


println(ks_results)
println("AIC_VALUES:",aic_results)

best_aic_idx = argmin(aic_results)
p_values = [pvalue(ks_result) for ks_result in ks_results]
best_ks_pval_idx = argmax(p_values)



# AIC Selection: Best Distribution Based on AIC
best_aic_dist = candidate_distributions[best_aic_idx]
println("Best Distribution based on AIC: ", best_aic_dist)

# KS Test Selection: Best Distribution Based on KS p-value
best_ks_dist = candidate_distributions[best_ks_pval_idx]
println("Best Distribution based on KS p-value: ", best_ks_dist)

# Plot 1: AIC-based selection
p1 = plot_with_distribution(selected_rvs, best_aic_dist, 
"AIC-selected Distribution", 
"AIC-Based Best Fit", "aic_best_fit_2.png")

# Plot 2: KS Test-based selection
p2 = plot_with_distribution(selected_rvs, best_ks_dist, 
"KS-selected Distribution", 
"KS Test-Based Best Fit", "ks_best_fit_2.png")

# Display the plots side by side
plot(p1, p2, layout = (1, 2), legend=:right, legendfontsize=6)

# Save the combined plot
savefig("combined_plot_2.png")


plot_kde(selected_rvs)


using DelimitedFiles

# Save the selected random variables to a CSV file
output_file = "selected_rvs.csv"
writedlm(output_file, selected_rvs', ',')  # Write the data as a row
println("Selected random variables saved to $output_file")


# After determining the best KS distribution (based on p-value)
best_ks_dist = candidate_distributions[best_ks_pval_idx]
println("Best Distribution based on KS p-value: ", best_ks_dist)


println("Enter the number of claims to select (n): ")
n = parse(Int, readline())  
    
# Step 3: Select n random claim sizes
selected_claims = select_random_claims(selected_rvs, n)
println("Randomly selected claim sizes: ", selected_claims)

sorted_claim_sizes = sort(selected_claims)

# Print the sorted list
println(sorted_claim_sizes)

