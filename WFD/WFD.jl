using LinearAlgebra
using Random

include("struct.jl")

"""
Computes node embeddings using weighted flow diffusion on an undirected weighted
graph G.

Inputs:
            G - Adjacency list representation of graph.
                Node indices must start from 1 and end with n, where n is the
                number of nodes in G.

            X - n-by-d matrix of node attributes.

         seed - Index of the seed node.

       source - Source mass assigned to the seed node.

    max_iters - Maximum number of passes for Random Permutation Coordinate
                Minimization. A single pass goes over all nodes that violate
                KKT conditions.

      epsilon - Tolerance on the maximum excess mass on nodes (which is
                equivalent to the maximum primal infeasibility).
                Diffusion process terminates whenever the excess mass is no
                greater than epsilon on all nodes.

Returns:
            x - Node embeddings.
                Applying sweepcut on x produces the final output cluster.
"""

function WFD(G::AdjacencyList, source::Vector{Float64}, sink::Vector{Float64};
            X=nothing, gamma::Float64=0.0, weights=nothing, max_iters::Int64=100,
            epsilon::Float64=1.0e-3)

    if weights === nothing
        weights = Dict{Int64,Vector{Float64}}()
        compute_edge_weight = true
    else
        compute_edge_weight = false
    end
    S = Set(findall(source.>0))
    mass = copy(source)
    x = zeros(Float64, G.nv)

    for t in 1:max_iters
        SP = [i for i in S if mass[i] > sink[i] + epsilon]
        if isempty(SP)
            break
        end
        for i in shuffle!(SP)
            if compute_edge_weight
                if !haskey(weights,i)
                    get_edge_weights!(G, X, i, weights, gamma)
                end
            end
            denom = sum(weights[i])
            push = (mass[i] - sink[i])/denom
            x[i] += push
            mass[i] = sink[i]
            for (idx,j) in enumerate(G.neighbors[i])
                if mass[j] == 0
                    push!(S,j)
                end
                mass[j] += weights[i][idx]*push
            end
        end
    end
    return x
end

function get_edge_weights!(G::AdjacencyList, X::Matrix{Float64}, i::Int64,
            weights::Dict{Int64,Vector{Float64}}, gamma::Float64)
    weights[i] = Vector{Float64}()
    for j in G.neighbors[i]
        push!(weights[i], exp(-gamma*norm(X[i,:]-X[j,:])^2))
    end
end

function connected_components(G::AdjacencyList)
    cc = Vector{Vector{Int64}}()
    covered_ids = zeros(Int64, G.nv)
    uncovered_ids = findall(iszero, covered_ids)
    while !isempty(uncovered_ids)
        connected_ids = zeros(Int64, G.nv)
        start_id = uncovered_ids[1]
        connected_ids[start_id] = 1
        boundary_ids = Int64[start_id]
        while !isempty(boundary_ids)
            search_ids = copy(boundary_ids)
            boundary_ids = Int64[]
            for i in search_ids
                for j in G.neighbors[i]
                    if connected_ids[j] == 0
                        push!(boundary_ids,j)
                        connected_ids[j] = 1
                    end
                end
            end
        end
        c = findall(!iszero, connected_ids)
        covered_ids[c] .= 1
        push!(cc, copy(c))
        uncovered_ids = findall(iszero, covered_ids)
    end
    return cc
end
