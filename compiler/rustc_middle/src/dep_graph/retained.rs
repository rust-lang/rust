use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::linked_graph::{Direction, INCOMING, LinkedGraph, NodeIndex};

use super::{DepNode, DepNodeIndex};

/// An in-memory copy of the current session's query dependency graph, which
/// is only enabled when `-Zquery-dep-graph` is set (for debugging/testing).
///
/// Normally, dependencies recorded during the current session are written to
/// disk and then forgotten, to avoid wasting memory on information that is
/// not needed when the compiler is working correctly.
pub struct RetainedDepGraph {
    pub inner: LinkedGraph<DepNode, ()>,
    pub indices: FxHashMap<DepNode, NodeIndex>,
}

impl RetainedDepGraph {
    pub fn new(prev_node_count: usize) -> Self {
        let node_count = prev_node_count + prev_node_count / 4;
        let edge_count = 6 * node_count;

        let inner = LinkedGraph::with_capacity(node_count, edge_count);
        let indices = FxHashMap::default();

        Self { inner, indices }
    }

    pub fn push(&mut self, index: DepNodeIndex, node: DepNode, edges: &[DepNodeIndex]) {
        let source = NodeIndex(index.as_usize());
        self.inner.add_node_with_idx(source, node);
        self.indices.insert(node, source);

        for &target in edges.iter() {
            self.inner.add_edge(source, NodeIndex(target.as_usize()), ());
        }
    }

    pub fn nodes(&self) -> Vec<&DepNode> {
        self.inner.all_nodes().iter().map(|n| n.data.as_ref().unwrap()).collect()
    }

    pub fn edges(&self) -> Vec<(&DepNode, &DepNode)> {
        self.inner
            .all_edges()
            .iter()
            .map(|edge| (edge.source(), edge.target()))
            .map(|(s, t)| (self.inner.node_data(s), self.inner.node_data(t)))
            .collect()
    }

    fn reachable_nodes(&self, node: &DepNode, direction: Direction) -> Vec<&DepNode> {
        if let Some(&index) = self.indices.get(node) {
            self.inner.depth_traverse(index, direction).map(|s| self.inner.node_data(s)).collect()
        } else {
            vec![]
        }
    }

    /// All nodes that can reach `node`.
    pub fn transitive_predecessors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.reachable_nodes(node, INCOMING)
    }
}
