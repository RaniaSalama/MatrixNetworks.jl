function bfs_test()
    A = load_matrix_network("bfs_example")
    bfs(MatrixNetwork(A),1)
    return true
end