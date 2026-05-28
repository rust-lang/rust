use itertools::Itertools;
use rustc_data_structures::graph;
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_index::Idx;
use rustc_middle::mir::coverage::Op;

use crate::coverage::counters::node_flow::{
    CounterTerm, NodeCounters, NodeFlowData, make_node_counters, node_flow_data_for_balanced_graph,
};

fn node_flow_data<G: graph::Successors>(graph: G) -> NodeFlowData<G::Node> {
    node_flow_data_for_balanced_graph(graph)
}

fn make_graph<Node: Idx + Ord>(num_nodes: usize, edge_pairs: Vec<(Node, Node)>) -> VecGraph<Node> {
    VecGraph::new(num_nodes, edge_pairs)
}

/// Example used in "Optimal Measurement Points for Program Frequency Counts"
/// (Knuth & Stevenson, 1973), but with 0-based node IDs.
#[test]
fn example_driver() {
    let graph = make_graph::<u32>(
        5,
        vec![(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 4), (3, 3), (3, 4), (4, 0)],
    );

    let node_flow_data = node_flow_data(&graph);
    let counters = make_node_counters(&node_flow_data, &[3, 1, 2, 0, 4]);

    assert_eq!(
        format_counter_expressions(&counters),
        &[
            // (comment to force vertical formatting for clarity)
            "[0]: +c0",
            "[1]: +c0 +c2 -c4",
            "[2]: +c2",
            "[3]: +c3",
            "[4]: +c4",
        ]
    );
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
        .counter_terms
        .indices()
        .map(|node| {
            let mut terms = counters.counter_terms[node].iter().collect::<Vec<_>>();
            terms.sort_by_key(|item| item.node.index());
            format!("[{node:?}]: {}", terms.into_iter().map(format_item).join(" "))
        })
        .collect()
}
