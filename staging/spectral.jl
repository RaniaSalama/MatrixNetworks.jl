

"""
Compute the Fiedler vector associated with the normalized Laplacian
of the graph with adjacency matrix A
"""
function fiedler_vector{V}(A::SparseMatrixCSC{V,Int})
    d = sum(A)
    # NOT DONE
end

"""
Compute a Fiedler PageRank vector 
"""
function fielder_pagerank{V}(A::SparseMatrixCSC{V,Int},seed::Dict{Int,V};
    alpha=0.99)

end

type PageRankProblem
    A::SparseMatrixCSC{Float64,Int}
    alpha::Float64
    seed::Int64
end    
    
function ppr_push(P::PageRankProblem, eps::Float64)
    # extract the sparse data structure
    colptr = P.A.colptr
    rowval = rowvals(P.A)
    n = size(P.A,1)
    
    # create the initial solution and residual
    x = Dict{Int,Float64}()
    r = Dict{Int,Float64}()
    
    r[P.seed] = 1.
    
    q = [P.seed]
    
    pushcount = 0
    pushvol = 0
    maxpush = 1./(eps*(1.-P.alpha))
    
    while length(q) > 0 && pushcount <= maxpush
        pushcount += 1
        u = shift!(q)
        du = Float64(colptr[u+1]-colptr[u])
        pushval = r[u] - 0.5*eps*du
        x[u] = get(x,u,0.0) + (1-P.alpha)*pushval
        r[u] = 0.5*eps*du
        
        pushval = pushval*P.alpha/du
        
        for nzi in colptr[u]:(colptr[u+1] - 1)
            pushvol += 1
            v = rowval[nzi]
            dv = Float64(colptr[v+1]-colptr[v]) # degree of v
            rvold = get(r,v,0.)
            rvnew = rvold + pushval
            r[v] = rvnew
            if rvnew > eps*dv && rvold <= eps*dv
                push!(q,v)
            end
        end
        pushvol += Int64(du)
    end
    
    return x, r, pushcount, pushvol
end    