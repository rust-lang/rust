use itertools::Itertools;
use rustc_data_structures::graph;
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_index::Idx;
use rustc_middle::mir::coverage::Op;

use super::{CounterTerm, MergedNodeFlowGraph, NodeCounters};

fn merged_node_flow_graph<G: graph::Successors>(graph: G) -> MergedNodeFlowGraph<G::Node> {
    MergedNodeFlowGraph::for_balanced_graph(graph)
}

fn make_graph<Node: Idx + Ord>(num_nodes: usize, edge_pairs: Vec<(Node, Node)>) -> VecGraph<Node> {
    VecGraph::new(num_nodes, edge_pairs)
}

/// Example used in "Optimal Measurement Points for Program Frequency Counts"
/// (Knuth & Stevenson, 1973), but with 0-based node IDs.
#[test]
fn example_driver() {
    let graph = make_graph::<u32>(5, vec![
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 4),
        (3, 3),
        (3, 4),
        (4, 0),
    ]);

    let merged = merged_node_flow_graph(&graph);
    let counters = merged.make_node_counters(&[3, 1, 2, 0, 4]);

    assert_eq!(format_counter_expressions(&counters), &[
        // (comment to force vertical formatting for clarity)
        "[0]: +c0",
        "[1]: +c0 +c2 -c4",
        "[2]: +c2",
        "[3]: +c3",
        "[4]: +c4",
    ]);
}

fn format_counter_expressions<Node: Idx>(counters: &NodeCounters<Node>) -> Vec<String> {
    let format_item = |&CounterTerm { node, op }| {
        let op = match op {
            Op::Subtract => '-',
            Op::Add => '+',
        };
        format!("{op}c{node:?}")
    };

    counters
        .counter_exprs
        .indices()
        .map(|node| {
            let mut expr = counters.counter_expr(node).iter().collect::<Vec<_>>();
            expr.sort_by_key(|item| item.node.index());
            format!("[{node:?}]: {}", expr.into_iter().map(format_item).join(" "))
        })
        .collect()
}
