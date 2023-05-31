using LinearAlgebra
include("struct.jl")


function edge_weights(G::AdjacencyList, X::Matrix{Float64}, gamma::Float64)
# Compute all edge weights for G
# The edge weights are aligned with G.neighbors

    weights = Vector{Vector{Float64}}()

    for i in 1:G.nv
        w = Vector{Float64}()
        for j in G.neighbors[i]
            push!(w, exp(-gamma*norm(X[i,:]-X[j,:])^2))
        end
        push!(weights, w)
    end

    return weights
end

function compute_conductance(G::AdjacencyList, C::Vector{Int64};
    weighted::Bool=true, edge_weight=nothing, attributes=nothing,
    gamma::Float64=0.0)

    if weighted
        if edge_weight === nothing
            edge_weight = edge_weights(G, attributes, gamma)
        end
        weighted_degree = [sum(w) for w in edge_weight]
        vol_G = sum(weighted_degree)
        vol_C = sum(weighted_degree[C])
    else
        vol_G = sum(G.degree)
        vol_C = sum(G.degree[C])
    end

    cut_C = 0
    set_C = Set(C)

    for i in C
        for (idx,j) in enumerate(G.neighbors[i])
            if !(j in set_C)
                if weighted
                    w_ij = edge_weight[i][idx]
                else
                    w_ij = 1.0
                end
                cut_C += w_ij
            end
        end
    end

    return cut_C, vol_C, cut_C/min(vol_C, vol_G - vol_C)
end
