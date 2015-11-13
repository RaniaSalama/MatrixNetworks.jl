
"""
Compute eigenvalues using direct calls to the ARPACK wrappers
to get type-stability.
"""
function _symeigs_smallest_arpack{V}(
            A::SparseMatrixCSC{V,Int},nev::Int,tol::V,v0::Vector{V})                        
            
    n::Int = Base.LinAlg.chksquare(A) # get the size
        
    # setup options 
    mode = 1
    sym = true
    iscmplx = false
    bmat = ByteString("I")
    ncv = max(2*nev,20)
    
    whichstr = ByteString("SA")
    maxiter = 300
    ritzvec = true
    sigma = 0.
    
    TOL = Array(V,1)
    TOL[1] = tol
    lworkl = ncv*(ncv + 8)
    v = Array(V, n, ncv)
    workd = Array(V, 3*n)
    workl = Array(V, lworkl)
    resid = Array(V, n)
    
    resid[:] = v0[:]
    
    info = zeros(Base.LinAlg.BlasInt, 1)
    info[1] = 1
    
    iparam = zeros(Base.LinAlg.BlasInt, 11)
    ipntr = zeros(Base.LinAlg.BlasInt, 11)
    ido = zeros(Base.LinAlg.BlasInt, 1)
    
    iparam[1] = Base.LinAlg.BlasInt(1)
    iparam[3] = Base.LinAlg.BlasInt(maxiter)
    iparam[7] = Base.LinAlg.BlasInt(mode)
    
    zernm1 = 0:(n-1)
    
    while true
        Base.LinAlg.ARPACK.saupd(
            ido, bmat, n, whichstr, nev, TOL, resid, ncv, v, n,
            iparam, ipntr, workd, workl, lworkl, info)
            
        load_idx = ipntr[1] + zernm1
        store_idx = ipntr[2] + zernm1
        
        x = workd[load_idx]
        
        if ido[1] == 1
            workd[store_idx] = A*x
        elseif ido[1] == 99
            break
        else
            error("unexpected ARPACK behavior")
        end
    end
    
    # calls to eupd
    howmny = ByteString("A")
    select = Array(Base.LinAlg.BlasInt, ncv)
    
    d = Array(V, nev)
    sigmar = ones(V,1)*sigma
    ldv = n
    Base.LinAlg.ARPACK.seupd(ritzvec, howmny, select, d, v, ldv, sigmar, 
        bmat, n, whichstr, nev, TOL, resid, ncv, v, ldv, 
        iparam, ipntr, workd, workl, lworkl, info)
    if info[1] != 0
        error("unexpected ARPACK exception")
    end
    
    p = sortperm(d)
    
    d = d[p]
    vectors = v[1:n,p]
    
    return (d,vectors,iparam[5])
end            

"""
Compute the Fiedler vector associated with the normalized Laplacian
of the graph with adjacency matrix A

The return vector is signed so that the number of positive
entries is at least the number of negative entries. This will
always give a

Returns
    (x,lam2)
"""
function fiedler_vector{V}(A::SparseMatrixCSC{V,Int};tol=1e-8,maxiter=300,dense=96)
    d = vec(sum(A,1))
    d = sqrt(d)
    n = size(A,1)
    if n > dense
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = full(L) + 2*eye(n)
        F = eigfact!(Symmetric(L))
        lam2 = F.values[2]-1.
        X = F.vectors
    else
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = L + 2*speye(n)
        ve = ones(n) # create a deterministic starting vector
        #(lams,X,nconv) = eigs(L;nev=2,which=:SR,tol=tol,v0=ve)
        (lams,X,nconv) = _symeigs_smallest_arpack(L,2,tol,ve)
        lam2 = lams[2]-1.
    end
    x1err = norm(X[:,1]*sign(X[1,1]) - d/norm(d))
    if x1err >= 10*tol
        warn(@sprintf("""
        the null-space vector associated with the normalized Laplacian
        was computed inaccurately (diff=%.3e); the Fiedler vector is
        probably wrong""",x1err))
    end

    x = vec(X[:,2])
    x = x./d # applied sqrt above

    # flip the sign if the number of pos. entries is less than the num of neg. entries
    nneg = sum(x .< 0.)
    if n-nneg < nneg
      x = -x;
    end

    return (x,lam2)
end

type RankedArray{S}
    data::S 
end

import Base: getindex, haskey

haskey{S}(A::RankedArray{S}, x::Int) = x >= 1 && x <= length(A.data)
getindex{S}(A::RankedArray{S}, i) = A.data[i]    

immutable SweepcutProfile{V,F}
    p::Vector{Int}
    conductance::Vector{F}
    cut::Vector{V}
    volume::Vector{V}
    
    function SweepcutProfile(p::Vector{Int}) 
        n = length(p)
        new(p,Array(F,n-1),Array(V,n-1),Array(V,n-1))
    end
end

"""
A - the sparse matrix representing the symmetric graph
p - the permutation vector
r - the rank of an item in the permutation vector
        p should be sorted in decreasing order so that
        i < j means that x[p[i]] < x[p[j]]
totalvol - the entire volume of the graph
maxvol - the maximum volume to consider, e.g. stop the sweep after maxvol

"""        
function sweepcut{V,T}(A::SparseMatrixCSC{V,Int}, p::Vector{Int}, r, 
    totalvol::V, maxvol::T)
    
    F = typeof(one(V)/one(V)) # find the conductance type
    output = SweepcutProfile{V,F}(p)
    
    nlast = length(p)
    
    cut = zero(V)
    vol = zero(V)
    colptr = A.colptr
    rowval = rowvals(A)
    nzval = A.nzval

    for (i,v) in enumerate(p)
        deltain = zero(V) # V might be pos. only... so delay subtraction
        deg = zero(V)
        rankv = getindex(r,v)
        for nzi in colptr[v]:(colptr[v+1] - 1)
            nbr = rowval[nzi] 
            deg += nzval[nzi]
            if haskey(r,nbr) # our neighbor is ranked
                if getindex(r,nbr) <= rankv # nbr is ranked lower, decrease cut
                    deltain += v == nbr ? nzval[nzi] : 2*nzval[nzi]
                end
            end
        end
        cut += deg
        cut -= deltain
        vol += deg
        
        # don't assign final values because they are unhelpful
        if i==nlast
            break
        end
        
        cond = cut/min(vol,totalvol-vol)
        output.conductance[i] = cond
        output.cut[i] = cut
        output.volume[i] = vol
    end
    @assert abs(cut) < 1e-12*totalvol
    return output
end    

function sweepcut{V,T}(A::SparseMatrixCSC{V,Int}, x::Vector{T}, vol::V) 
    p = sortperm(x,rev=true)
    ranks = Array(Int, length(x))
    for (i,v) in enumerate(p)
        ranks[v] = i
    end
    r = RankedArray(ranks)
    return sweepcut(A, p, r, vol, Inf)
end    

sweepcut{V,T}(A::SparseMatrixCSC{V,Int}, x::Vector{T}) = 
    sweepcut(A, x, sum(A))

#sweepcut{V,F}(A::SparseMatrixCSC{V,Int}, x::SparseMatrixCSC{F,Int}) = sweepcut(A, x, sum(A)) 


 
"""
Compute a Fiedler PageRank vector
"""
function fielder_pagerank{V}(A::SparseMatrixCSC{V,Int},seed::Dict{Int,V};alpha=0.99)

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
