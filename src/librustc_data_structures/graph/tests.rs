// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use graph::*;
use std::fmt::Debug;

type TestGraph = Graph<&'static str, &'static str>;

fn create_graph() -> TestGraph {
    let mut graph = Graph::new();

    // Create a simple graph
    //
    //          F
    //          |
    //          V
    //    A --> B --> C
    //          |     ^
    //          v     |
    //          D --> E

    let a = graph.add_node("A");
    let b = graph.add_node("B");
    let c = graph.add_node("C");
    let d = graph.add_node("D");
    let e = graph.add_node("E");
    let f = graph.add_node("F");

    graph.add_edge(a, b, "AB");
    graph.add_edge(b, c, "BC");
    graph.add_edge(b, d, "BD");
    graph.add_edge(d, e, "DE");
    graph.add_edge(e, c, "EC");
    graph.add_edge(f, b, "FB");

    return graph;
}

fn create_graph_with_cycle() -> TestGraph {
    let mut graph = Graph::new();

    // Create a graph with a cycle.
    //
    //    A --> B <-- +
    //          |     |
    //          v     |
    //          C --> D

    let a = graph.add_node("A");
    let b = graph.add_node("B");
    let c = graph.add_node("C");
    let d = graph.add_node("D");

    graph.add_edge(a, b, "AB");
    graph.add_edge(b, c, "BC");
    graph.add_edge(c, d, "CD");
    graph.add_edge(d, b, "DB");

    return graph;
}

#[test]
fn each_node() {
    let graph = create_graph();
    let expected = ["A", "B", "C", "D", "E", "F"];
    graph.each_node(|idx, node| {
        assert_eq!(&expected[idx.0], graph.node_data(idx));
        assert_eq!(expected[idx.0], node.data);
        true
    });
}

#[test]
fn each_edge() {
    let graph = create_graph();
    let expected = ["AB", "BC", "BD", "DE", "EC", "FB"];
    graph.each_edge(|idx, edge| {
        assert_eq!(&expected[idx.0], graph.edge_data(idx));
        assert_eq!(expected[idx.0], edge.data);
        true
    });
}

fn test_adjacent_edges<N: PartialEq + Debug, E: PartialEq + Debug>(graph: &Graph<N, E>,
                                                                   start_index: NodeIndex,
                                                                   start_data: N,
                                                                   expected_incoming: &[(E, N)],
                                                                   expected_outgoing: &[(E, N)]) {
    assert!(graph.node_data(start_index) == &start_data);

    let mut counter = 0;
    for (edge_index, edge) in graph.incoming_edges(start_index) {
        assert!(graph.edge_data(edge_index) == &edge.data);
        assert!(counter < expected_incoming.len());
        debug!("counter={:?} expected={:?} edge_index={:?} edge={:?}",
               counter,
               expected_incoming[counter],
               edge_index,
               edge);
        match expected_incoming[counter] {
            (ref e, ref n) => {
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
        assert!(graph.edge_data(edge_index) == &edge.data);
        assert!(counter < expected_outgoing.len());
        debug!("counter={:?} expected={:?} edge_index={:?} edge={:?}",
               counter,
               expected_outgoing[counter],
               edge_index,
               edge);
        match expected_outgoing[counter] {
            (ref e, ref n) => {
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
    test_adjacent_edges(&graph, NodeIndex(0), "A", &[], &[("AB", "B")]);
}

#[test]
fn each_adjacent_from_b() {
    let graph = create_graph();
    test_adjacent_edges(&graph,
                        NodeIndex(1),
                        "B",
                        &[("FB", "F"), ("AB", "A")],
                        &[("BD", "D"), ("BC", "C")]);
}

#[test]
fn each_adjacent_from_c() {
    let graph = create_graph();
    test_adjacent_edges(&graph, NodeIndex(2), "C", &[("EC", "E"), ("BC", "B")], &[]);
}

#[test]
fn each_adjacent_from_d() {
    let graph = create_graph();
    test_adjacent_edges(&graph, NodeIndex(3), "D", &[("BD", "B")], &[("DE", "E")]);
}

#[test]
fn is_node_cyclic_a() {
    let graph = create_graph_with_cycle();
    assert!(!graph.is_node_cyclic(NodeIndex(0)));
}

#[test]
fn is_node_cyclic_b() {
    let graph = create_graph_with_cycle();
    assert!(graph.is_node_cyclic(NodeIndex(1)));
}
