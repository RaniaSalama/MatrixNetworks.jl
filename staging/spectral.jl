

"""
Compute the Fiedler vector associated with the normalized Laplacian
of the graph with adjacency matrix A

Returns
    (x,lam2) 
"""
function fiedler_vector{V}(A::SparseMatrixCSC{V,Int};tol=1e-8)
    d = sum(A)
    n = size(A)
    [ai,aj,av] = findnz(A)
    L = sparse(ai,aj,av./(sqrt(d[ai].*d[aj])),n,n)
    L = L + speye(n)
    d,V = eigs(L;nev=2,which=:SR,tol=tol)
    lam2 = d[2]-1.
    
    x = V[:,2]
    x = x./sqrt(d)
    return (x,lam2)
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