use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::linked_graph::{Direction, INCOMING, LinkedGraph};

use super::{DepNode, DepNodeIndex};

pub struct DepGraphQuery {
    pub graph: LinkedGraph<DepNodeIndex, DepNode, ()>,
    pub indices: FxHashMap<DepNode, DepNodeIndex>,
}

impl DepGraphQuery {
    pub fn new(prev_node_count: usize) -> DepGraphQuery {
        let node_count = prev_node_count + prev_node_count / 4;
        let edge_count = 6 * node_count;

        let graph = LinkedGraph::with_capacity(node_count, edge_count);
        let indices = FxHashMap::default();

        DepGraphQuery { graph, indices }
    }

    pub fn push(&mut self, index: DepNodeIndex, node: DepNode, edges: &[DepNodeIndex]) {
        self.graph.add_node(index, node);
        self.indices.insert(node, index);

        for &target in edges.iter() {
            self.graph.add_edge(index, target, ());
        }
    }

    pub fn nodes(&self) -> Vec<&DepNode> {
        self.graph.nodes_iter().map(|n| &n.data).collect()
    }

    pub fn edges(&self) -> Vec<(&DepNode, &DepNode)> {
        self.graph
            .all_edges()
            .iter()
            .map(|edge| (edge.source(), edge.target()))
            .map(|(s, t)| (self.graph.node_data(s), self.graph.node_data(t)))
            .collect()
    }

    fn reachable_nodes(&self, node: &DepNode, direction: Direction) -> Vec<&DepNode> {
        if let Some(&index) = self.indices.get(node) {
            self.graph.depth_traverse(index, direction).map(|s| self.graph.node_data(s)).collect()
        } else {
            vec![]
        }
    }

    /// All nodes that can reach `node`.
    pub fn transitive_predecessors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.reachable_nodes(node, INCOMING)
    }
}
