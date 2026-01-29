use rustc_index::Idx;
use tracing::debug;

use super::{Debug, LinkedGraph, FrozenLinkedGraph};

type TestGraph = FrozenLinkedGraph<usize, &'static str, &'static str>;

fn create_graph() -> TestGraph {
    let mut graph = LinkedGraph::new();

    // Create a simple graph
    //
    //          F
    //          |
    //          V
    //    A --> B --> C
    //          |     ^
    //          v     |
    //          D --> E

    let a = 0; graph.add_node(a, "A");
    let b = 1; graph.add_node(b, "B");
    let c = 2; graph.add_node(c, "C");
    let d = 3; graph.add_node(d, "D");
    let e = 4; graph.add_node(e, "E");
    let f = 5; graph.add_node(f, "F");

    graph.add_edge(a, b, "AB");
    graph.add_edge(b, c, "BC");
    graph.add_edge(b, d, "BD");
    graph.add_edge(d, e, "DE");
    graph.add_edge(e, c, "EC");
    graph.add_edge(f, b, "FB");

    return graph.freeze();
}

#[test]
fn each_node() {
    let graph = create_graph();
    let expected = ["A", "B", "C", "D", "E", "F"];
    graph.each_node(|idx, node| {
        assert_eq!(&expected[idx], graph.node_data(idx));
        assert_eq!(expected[idx], node.data);
        true
    });
}

#[test]
fn each_edge() {
    let graph = create_graph();
    let expected = ["AB", "BC", "BD", "DE", "EC", "FB"];
    graph.each_edge(|idx, edge| {
        assert_eq!(expected[idx.0], edge.data);
        true
    });
}

fn test_adjacent_edges<I: Idx, N: PartialEq + Debug, E: PartialEq + Debug>(
    graph: &FrozenLinkedGraph<I, N, E>,
    start_index: I,
    start_data: N,
    expected_incoming: &[(E, N)],
    expected_outgoing: &[(E, N)],
) {
    assert!(graph.node_data(start_index) == &start_data);

    let mut counter = 0;
    for (edge_index, edge) in graph.incoming_edges(start_index) {
        assert!(counter < expected_incoming.len());
        debug!(
            "counter={:?} expected={:?} edge_index={:?} edge={:?}",
            counter, expected_incoming[counter], edge_index, edge
        );
        match &expected_incoming[counter] {
            (e, n) => {
                assert!(e == &edge.data);
                assert!(n == graph.node_data(edge.source()));
                assert!(start_index == edge.target);
            }
        }
        counter += 1;
    }
    assert_eq!(counter, expected_incoming.len());

    let mut counter = 0;
    for (edge_index, edge) in graph.outgoing_edges(start_index) {
        assert!(counter < expected_outgoing.len());
        debug!(
            "counter={:?} expected={:?} edge_index={:?} edge={:?}",
            counter, expected_outgoing[counter], edge_index, edge
        );
        match &expected_outgoing[counter] {
            (e, n) => {
                assert!(e == &edge.data);
                assert!(start_index == edge.source);
                assert!(n == graph.node_data(edge.target));
            }
        }
        counter += 1;
    }
    assert_eq!(counter, expected_outgoing.len());
}

#[test]
fn each_adjacent_from_a() {
    let graph = create_graph();
    test_adjacent_edges(&graph, 0, "A", &[], &[("AB", "B")]);
}

#[test]
fn each_adjacent_from_b() {
    let graph = create_graph();
    test_adjacent_edges(&graph, 1, "B", &[("FB", "F"), ("AB", "A")], &[("BD", "D"), ("BC", "C")]);
}

#[test]
fn each_adjacent_from_c() {
    let graph = create_graph();
    test_adjacent_edges(&graph, 2, "C", &[("EC", "E"), ("BC", "B")], &[]);
}

#[test]
fn each_adjacent_from_d() {
    let graph = create_graph();
    test_adjacent_edges(&graph, 3, "D", &[("BD", "B")], &[("DE", "E")]);
}
