# MatrixNetworks

## To install package:
Pkg.clone("https://github.com/nassarhuda/MatrixNetworks.jl.git")

using MatrixNetworks
## To be able to see documentation make sure package Lexicon is installed:
Pkg.add("Lexicon")

using Lexicon

## Example
? bfs

? bipartite_matching

## To run test cases:
Pkg.test("MatrixNetworks")

## Some examples:
### clustercoeffs
```
file_path = Pkg.dir("MatrixNetworks/data/clique-10.smat")
A = readSMAT(file_path)
cc = clustercoeffs(MatrixNetwork(A))
```

### bipartite_matching:
```
ei = [1;2;3]
ej = [3;2;4]
M_out = bipartite_matching([10;12;13],ei,ej)\n
M_out.weight\n
M_out.cardinality\n
M_out.match\n
MatrixNetworks.create_sparse(bipartite_matching(W)) # get the sparse matrix\n
MatrixNetworks.edge_list(bipartite_matching(W)) # get the edgelist\n
MatrixNetworks.edge_indicator(M_out,ei,ej)
```

