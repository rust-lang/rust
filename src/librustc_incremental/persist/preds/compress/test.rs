// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;

fn reduce(graph: &Graph<&'static str, ()>,
          inputs: &[&'static str],
          outputs: &[&'static str],
          expected: &[&'static str])
{
    let reduce = GraphReduce::new(&graph,
                                  |n| inputs.contains(n),
                                  |n| outputs.contains(n));
    let result = reduce.compute();
    let mut edges: Vec<String> =
        result.graph
              .all_edges()
              .iter()
              .map(|edge| format!("{} -> {}",
                                  result.graph.node_data(edge.source()),
                                  result.graph.node_data(edge.target())))
              .collect();
    edges.sort();
    println!("{:#?}", edges);
    assert_eq!(edges.len(), expected.len());
    for (expected, actual) in expected.iter().zip(&edges) {
        assert_eq!(expected, actual);
    }
}

#[test]
fn test1() {
    //  +---------------+
    //  |               |
    //  |      +--------|------+
    //  |      |        v      v
    // [A] -> [C0] -> [C1]    [D]
    //        [  ] <- [  ] -> [E]
    //                  ^
    // [B] -------------+
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        C1 -> E,
    };

    // [A] -> [C1] -> [D]
    // [B] -> [  ] -> [E]
    reduce(&graph, &["A", "B"], &["D", "E"], &[
        "A -> C1",
        "B -> C1",
        "C1 -> D",
        "C1 -> E",
    ]);
}

#[test]
fn test2() {
    //  +---------------+
    //  |               |
    //  |      +--------|------+
    //  |      |        v      v
    // [A] -> [C0] -> [C1]    [D] -> [E]
    //        [  ] <- [  ]
    //                  ^
    // [B] -------------+
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        D -> E,
    };

    // [A] -> [D] -> [E]
    // [B] -> [ ]
    reduce(&graph, &["A", "B"], &["D", "E"], &[
        "A -> D",
        "B -> D",
        "D -> E",
    ]);
}

#[test]
fn test2b() {
    // Variant on test2 in which [B] is not
    // considered an input.
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        D -> E,
    };

    // [A] -> [D] -> [E]
    reduce(&graph, &["A"], &["D", "E"], &[
        "A -> D",
        "D -> E",
    ]);
}

#[test]
fn test3() {

    // Edges going *downwards*, so 0, 1 and 2 are inputs,
    // while 7, 8, and 9 are outputs.
    //
    //     0     1   2
    //     |      \ /
    //     3---+   |
    //     |   |   |
    //     |   |   |
    //     4   5   6
    //      \ / \ / \
    //       |   |   |
    //       7   8   9
    //
    // Here the end result removes node 4, instead encoding an edge
    // from n3 -> n7, but keeps nodes 5 and 6, as they are common
    // inputs to nodes 8/9.

    let (graph, _nodes) = graph! {
        n0 -> n3,
        n3 -> n4,
        n3 -> n5,
        n4 -> n7,
        n5 -> n7,
        n5 -> n8,
        n1 -> n6,
        n2 -> n6,
        n6 -> n8,
        n6 -> n9,
    };

    reduce(&graph, &["n0", "n1", "n2"], &["n7", "n8", "n9"], &[
        "n0 -> n3",
        "n1 -> n6",
        "n2 -> n6",
        "n3 -> n5",
        "n3 -> n7",
        "n5 -> n7",
        "n5 -> n8",
        "n6 -> n8",
        "n6 -> n9"
    ]);
}

#[test]
fn test_cached_dfs_cyclic() {

    //    0       1 <---- 2       3
    //    ^       |       ^       ^
    //    |       v       |       |
    //    4 ----> 5 ----> 6 ----> 7
    //    ^       ^       ^       ^
    //    |       |       |       |
    //    8       9      10      11

    let (graph, _nodes) = graph! {
        // edges from above diagram, in columns, top-to-bottom:
        n4 -> n0,
        n8 -> n4,
        n4 -> n5,
        n1 -> n5,
        n9 -> n5,
        n2 -> n1,
        n5 -> n6,
        n6 -> n2,
        n10 -> n6,
        n6 -> n7,
        n7 -> n3,
        n11 -> n7,
    };

    //    0       1  2            3
    //    ^       ^ /             ^
    //    |       |/              |
    //    4 ----> 5 --------------+
    //    ^       ^ \             |
    //    |       |  \            |
    //    8       9   10         11

    reduce(&graph, &["n8", "n9", "n10", "n11"], &["n0", "n1", "n2", "n3"], &[
        "n10 -> n5",
        "n11 -> n3",
        "n4 -> n0",
        "n4 -> n5",
        "n5 -> n1",
        "n5 -> n2",
        "n5 -> n3",
        "n8 -> n4",
        "n9 -> n5"
    ]);
}

/// Demonstrates the case where we don't reduce as much as we could.
#[test]
fn suboptimal() {
    let (graph, _nodes) = graph! {
        INPUT0 -> X,
        X -> OUTPUT0,
        X -> OUTPUT1,
    };

    reduce(&graph, &["INPUT0"], &["OUTPUT0", "OUTPUT1"], &[
        "INPUT0 -> X",
        "X -> OUTPUT0",
        "X -> OUTPUT1"
    ]);
}

#[test]
fn test_cycle_output() {
    //  +---------------+
    //  |               |
    //  |      +--------|------+
    //  |      |        v      v
    // [A] -> [C0] <-> [C1] <- [D]
    //                  +----> [E]
    //                          ^
    // [B] ----------------- ---+
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> E,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        C1 -> E,
        D -> C1,
    };

    // [A] -> [C0] --> [D]
    //          +----> [E]
    //                  ^
    // [B] -------------+
    reduce(&graph, &["A", "B"], &["D", "E"], &[
        "A -> C0",
        "B -> E",
        "C0 -> D",
        "C0 -> E",
    ]);
}
