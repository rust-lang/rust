use super::*;

fn create_graph() -> VecGraph<usize> {
    // Create a simple graph
    //
    //          5
    //          |
    //          V
    //    0 --> 1 --> 2
    //          |
    //          v
    //          3 --> 4
    //
    //    6

    VecGraph::new(
        7,
        vec![
            (0, 1),
            (1, 2),
            (1, 3),
            (3, 4),
            (5, 1),
        ],
    )
}

#[test]
fn num_nodes() {
    let graph = create_graph();
    assert_eq!(graph.num_nodes(), 7);
}

#[test]
fn succesors() {
    let graph = create_graph();
    assert_eq!(graph.successors(0), &[1]);
    assert_eq!(graph.successors(1), &[2, 3]);
    assert_eq!(graph.successors(2), &[]);
    assert_eq!(graph.successors(3), &[4]);
    assert_eq!(graph.successors(4), &[]);
    assert_eq!(graph.successors(5), &[1]);
    assert_eq!(graph.successors(6), &[]);
}

#[test]
fn dfs() {
    let graph = create_graph();
    let dfs: Vec<_> = graph.depth_first_search(0).collect();
    assert_eq!(dfs, vec![0, 1, 3, 4, 2]);
}
