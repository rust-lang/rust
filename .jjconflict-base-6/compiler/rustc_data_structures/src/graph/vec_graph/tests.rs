use super::*;
use crate::graph;

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

    VecGraph::new(7, vec![(0, 1), (1, 2), (1, 3), (3, 4), (5, 1)])
}

fn create_graph_with_back_refs() -> VecGraph<usize, true> {
    // Same as above
    VecGraph::new(7, vec![(0, 1), (1, 2), (1, 3), (3, 4), (5, 1)])
}

#[test]
fn num_nodes() {
    let graph = create_graph();
    assert_eq!(graph.num_nodes(), 7);

    let graph = create_graph_with_back_refs();
    assert_eq!(graph.num_nodes(), 7);
}

#[test]
fn successors() {
    let graph = create_graph();
    assert_eq!(graph.successors(0), &[1]);
    assert_eq!(graph.successors(1), &[2, 3]);
    assert_eq!(graph.successors(2), &[] as &[usize]);
    assert_eq!(graph.successors(3), &[4]);
    assert_eq!(graph.successors(4), &[] as &[usize]);
    assert_eq!(graph.successors(5), &[1]);
    assert_eq!(graph.successors(6), &[] as &[usize]);

    let graph = create_graph_with_back_refs();
    assert_eq!(graph.successors(0), &[1]);
    assert_eq!(graph.successors(1), &[2, 3]);
    assert_eq!(graph.successors(2), &[] as &[usize]);
    assert_eq!(graph.successors(3), &[4]);
    assert_eq!(graph.successors(4), &[] as &[usize]);
    assert_eq!(graph.successors(5), &[1]);
    assert_eq!(graph.successors(6), &[] as &[usize]);
}

#[test]
fn predecessors() {
    let graph = create_graph_with_back_refs();
    assert_eq!(graph.predecessors(0), &[]);
    assert_eq!(graph.predecessors(1), &[0, 5]);
    assert_eq!(graph.predecessors(2), &[1]);
    assert_eq!(graph.predecessors(3), &[1]);
    assert_eq!(graph.predecessors(4), &[3]);
    assert_eq!(graph.predecessors(5), &[]);
    assert_eq!(graph.predecessors(6), &[]);
}

#[test]
fn dfs() {
    let graph = create_graph();
    let dfs: Vec<_> = graph::depth_first_search(&graph, 0).collect();
    assert_eq!(dfs, vec![0, 1, 3, 4, 2]);

    let graph = create_graph_with_back_refs();
    let dfs: Vec<_> = graph::depth_first_search(&graph, 0).collect();
    assert_eq!(dfs, vec![0, 1, 3, 4, 2]);
}
