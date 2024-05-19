use super::super::tests::TestGraph;

use super::*;

#[test]
fn diamond_post_order() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);

    let result = post_order_from(&graph, 0);
    assert_eq!(result, vec![3, 1, 2, 0]);
}

#[test]
fn is_cyclic() {
    use super::super::is_cyclic;

    let diamond_acyclic = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let diamond_cyclic = TestGraph::new(0, &[(0, 1), (1, 2), (2, 3), (3, 0)]);

    assert!(!is_cyclic(&diamond_acyclic));
    assert!(is_cyclic(&diamond_cyclic));
}

#[test]
fn dfs() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)]);

    let result: Vec<usize> = DepthFirstSearch::new(&graph).with_start_node(0).collect();
    assert_eq!(result, vec![0, 2, 3, 1]);
}

#[test]
fn dfs_debug() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)]);
    let mut dfs = DepthFirstSearch::new(&graph).with_start_node(0);
    dfs.complete_search();
    assert_eq!(format!("{{0, 1, 2, 3}}"), format!("{:?}", dfs));
}
