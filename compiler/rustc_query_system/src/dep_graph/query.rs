use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::implementation::{Direction, Graph, NodeIndex, INCOMING};
use rustc_index::vec::IndexVec;

use super::{DepNode, DepNodeIndex};

pub struct DepGraphQuery {
    pub graph: Graph<DepNode, ()>,
    pub indices: FxHashMap<DepNode, NodeIndex>,
    pub dep_index_to_index: IndexVec<DepNodeIndex, Option<NodeIndex>>,
}

impl DepGraphQuery {
    pub fn new(prev_node_count: usize) -> DepGraphQuery {
        let node_count = prev_node_count + prev_node_count / 4;
        let edge_count = 6 * node_count;

        let graph = Graph::with_capacity(node_count, edge_count);
        let indices = FxHashMap::default();
        let dep_index_to_index = IndexVec::new();

        DepGraphQuery { graph, indices, dep_index_to_index }
    }

    pub fn push(&mut self, index: DepNodeIndex, node: DepNode, edges: &[DepNodeIndex]) {
        let source = self.graph.add_node(node);
        if index.index() >= self.dep_index_to_index.len() {
            self.dep_index_to_index.resize(index.index() + 1, None);
        }
        self.dep_index_to_index[index] = Some(source);
        self.indices.insert(node, source);

        for &target in edges.iter() {
            let target = self.dep_index_to_index[target];
            // We may miss the edges that are pushed while the `DepGraphQuery` is being accessed.
            // Skip them to issues.
            if let Some(target) = target {
                self.graph.add_edge(source, target, ());
            }
        }
    }

    pub fn nodes(&self) -> Vec<&DepNode> {
        self.graph.all_nodes().iter().map(|n| &n.data).collect()
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
