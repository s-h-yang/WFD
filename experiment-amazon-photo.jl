using Random
using Printf
using DelimitedFiles
using Statistics
using LinearAlgebra

include("WFD/WFD.jl")
include("WFD/common.jl")

gamma = 0.02

dataset = "amazon_photo"

labels = vec(readdlm("data/"*dataset*"/labels.txt", ' ', Int64, '\n'))
ll = Vector{Vector{Int64}}()
for line in readlines("data/"*dataset*"/adj.txt")
    if isempty(strip(line))
        push!(ll,[])
        continue
    end
    push!(ll,[parse(Int64, i) for i in split(strip(line), " ")])
end
degree = [length(l) for l in ll]
n = length(ll)
degree = [length(l) for l in ll]
G = AdjacencyList(ll, degree, n)

cc = connected_components(G)
for i in 2:length(cc)
    labels[cc[i]] .= -1
end 

Z = zeros(n,1)
X = readdlm("data/"*dataset*"/attributes_gc.txt", ' ', Float64, '\n')

edge_weight_z = edge_weights(G, Z, 0.)
edge_weight_x = edge_weights(G, X, gamma)

num_classes = maximum(labels)
BASE_results = Dict{Int64,Vector{Float64}}(i=>[] for i in 1:num_classes)
WFD_results = Dict{Int64,Vector{Float64}}(i=>[] for i in 1:num_classes)

sink = [Float64(d) for d in G.degree]

for c in 1:num_classes
    
    K = findall(x->x==c, labels)
    k = length(K)
    vol_K = sum(G.degree[K])
    
    for i = 1:min(100,length(K))
        
        s = K[i]
        
        selected_f1_base = 0
        selected_f1_wfd = 0
        best_cond_base = 1
        best_cond_wfd = 1
        
        for alpha = 1.5:0.25:5
            
            source = zeros(n)
            source[s] = alpha*vol_K
            
            x = WFD(G, source, sink, weights=edge_weight_z, max_iters=100, epsilon=1.0e-4)
            C = findall(!iszero,x)
            pr = length(intersect(Set(C),Set(K)))/length(C)
            re = length(intersect(Set(C),Set(K)))/k
            f1 = 2*pr*re/(pr+re)
            _, _, conductance = compute_conductance(G, C, weighted=false)
            if conductance < best_cond_base
                best_cond_base = conductance
                selected_f1_base = f1
            end
            
            x = WFD(G, source, sink, weights=edge_weight_x, max_iters=100, epsilon=1.0e-4)
            C = findall(!iszero,x)
            pr = length(intersect(Set(C),Set(K)))/length(C)
            re = length(intersect(Set(C),Set(K)))/k
            f1 = 2*pr*re/(pr+re)
            _, _, conductance = compute_conductance(G, C, edge_weight=edge_weight_x)
            if conductance < best_cond_wfd
                best_cond_wfd = conductance
                selected_f1_wfd = f1
            end
        end

        @printf("Class %d, seed %d, base %.3f, WFD %.3f\n", c, i, selected_f1_base, selected_f1_wfd)
        flush(stdout)
        
        push!(BASE_results[c], selected_f1_base)
        push!(WFD_results[c], selected_f1_wfd)
    end
end


open("F1-amazon-photo-base.txt", "w") do f
    for i in 1:num_classes
        @printf(f, "%.4f\n", mean(BASE_results[i]))
    end
end

open("F1-amazon-photo-wfd.txt", "w") do f
    for i in 1:num_classes
        @printf(f, "%.4f\n", mean(WFD_results[i]))
    end
end
