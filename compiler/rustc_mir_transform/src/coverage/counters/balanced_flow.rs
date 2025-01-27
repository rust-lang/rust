//! A control-flow graph can be said to have “balanced flow” if the flow
//! (execution count) of each node is equal to the sum of its in-edge flows,
//! and also equal to the sum of its out-edge flows.
//!
//! Control-flow graphs typically have one or more nodes that don't satisfy the
//! balanced-flow property, e.g.:
//! - The start node has out-edges, but no in-edges.
//! - Return nodes have in-edges, but no out-edges.
//! - `Yield` nodes can have an out-flow that is less than their in-flow.
//! - Inescapable loops cause the in-flow/out-flow relationship to break down.
//!
//! Balanced-flow graphs are nevertheless useful for analysis, so this module
//! provides a wrapper type ([`BalancedFlowGraph`]) that imposes balanced flow
//! on an underlying graph. This is done by non-destructively adding synthetic
//! nodes and edges as necessary.

use rustc_data_structures::graph;
use rustc_data_structures::graph::iterate::DepthFirstSearch;
use rustc_data_structures::graph::reversed::ReversedGraph;
use rustc_index::Idx;
use rustc_index::bit_set::DenseBitSet;

/// A view of an underlying graph that has been augmented to have “balanced flow”.
/// This means that the flow (execution count) of each node is equal to the
/// sum of its in-edge flows, and also equal to the sum of its out-edge flows.
///
/// To achieve this, a synthetic "sink" node is non-destructively added to the
/// graph, with synthetic in-edges from these nodes:
/// - Any node that has no out-edges.
/// - Any node that explicitly requires a sink edge, as indicated by a
///   caller-supplied `force_sink_edge` function.
/// - Any node that would otherwise be unable to reach the sink, because it is
///   part of an inescapable loop.
///
/// To make the graph fully balanced, there is also a synthetic edge from the
/// sink node back to the start node.
///
/// ---
/// The benefit of having a balanced-flow graph is that it can be subsequently
/// transformed in ways that are guaranteed to preserve balanced flow
/// (e.g. merging nodes together), which is useful for discovering relationships
/// between the node flows of different nodes in the graph.
pub(crate) struct BalancedFlowGraph<G: graph::DirectedGraph> {
    graph: G,
    sink_edge_nodes: DenseBitSet<G::Node>,
    pub(crate) sink: G::Node,
}

impl<G: graph::DirectedGraph> BalancedFlowGraph<G> {
    /// Creates a balanced view of an underlying graph, by adding a synthetic
    /// sink node that has in-edges from nodes that need or request such an edge,
    /// and a single out-edge to the start node.
    ///
    /// Assumes that all nodes in the underlying graph are reachable from the
    /// start node.
    pub(crate) fn for_graph(graph: G, force_sink_edge: impl Fn(G::Node) -> bool) -> Self
    where
        G: graph::ControlFlowGraph,
    {
        let mut sink_edge_nodes = DenseBitSet::new_empty(graph.num_nodes());
        let mut dfs = DepthFirstSearch::new(ReversedGraph::new(&graph));

        // First, determine the set of nodes that explicitly request or require
        // an out-edge to the sink.
        for node in graph.iter_nodes() {
            if force_sink_edge(node) || graph.successors(node).next().is_none() {
                sink_edge_nodes.insert(node);
                dfs.push_start_node(node);
            }
        }

        // Next, find all nodes that are currently not reverse-reachable from
        // `sink_edge_nodes`, and add them to the set as well.
        dfs.complete_search();
        sink_edge_nodes.union_not(dfs.visited_set());

        // The sink node is 1 higher than the highest real node.
        let sink = G::Node::new(graph.num_nodes());

        BalancedFlowGraph { graph, sink_edge_nodes, sink }
    }
}

impl<G> graph::DirectedGraph for BalancedFlowGraph<G>
where
    G: graph::DirectedGraph,
{
    type Node = G::Node;

    /// Returns the number of nodes in this balanced-flow graph, which is 1
    /// more than the number of nodes in the underlying graph, to account for
    /// the synthetic sink node.
    fn num_nodes(&self) -> usize {
        // The sink node's index is already the size of the underlying graph,
        // so just add 1 to that instead.
        self.sink.index() + 1
    }
}

impl<G> graph::StartNode for BalancedFlowGraph<G>
where
    G: graph::StartNode,
{
    fn start_node(&self) -> Self::Node {
        self.graph.start_node()
    }
}

impl<G> graph::Successors for BalancedFlowGraph<G>
where
    G: graph::StartNode + graph::Successors,
{
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        let real_edges;
        let sink_edge;

        if node == self.sink {
            // The sink node has no real out-edges, and one synthetic out-edge
            // to the start node.
            real_edges = None;
            sink_edge = Some(self.graph.start_node());
        } else {
            // Real nodes have their real out-edges, and possibly one synthetic
            // out-edge to the sink node.
            real_edges = Some(self.graph.successors(node));
            sink_edge = self.sink_edge_nodes.contains(node).then_some(self.sink);
        }

        real_edges.into_iter().flatten().chain(sink_edge)
    }
}
