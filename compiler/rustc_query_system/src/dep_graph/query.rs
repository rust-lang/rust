use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::implementation::{Direction, Graph, NodeIndex, INCOMING};

use super::{DepKind, DepNode, DepNodeIndex};

pub struct DepGraphQuery<K> {
    pub graph: Graph<DepNode<K>, ()>,
    pub indices: FxHashMap<DepNode<K>, NodeIndex>,
}

impl<K: DepKind> DepGraphQuery<K> {
    pub fn new(prev_node_count: usize) -> DepGraphQuery<K> {
        let node_count = prev_node_count + prev_node_count / 4;
        let edge_count = 6 * node_count;

        let graph = Graph::with_capacity(node_count, edge_count);
        let indices = FxHashMap::default();

        DepGraphQuery { graph, indices }
    }

    pub fn push(&mut self, index: DepNodeIndex, node: DepNode<K>, edges: &[DepNodeIndex]) {
        let source = self.graph.add_node(node);
        debug_assert_eq!(index.index(), source.0);
        self.indices.insert(node, source);

        for &target in edges.iter() {
            let target = NodeIndex(target.index());
            self.graph.add_edge(source, target, ());
        }
    }

    pub fn nodes(&self) -> Vec<&DepNode<K>> {
        self.graph.all_nodes().iter().map(|n| &n.data).collect()
    }

    pub fn edges(&self) -> Vec<(&DepNode<K>, &DepNode<K>)> {
        self.graph
            .all_edges()
            .iter()
            .map(|edge| (edge.source(), edge.target()))
            .map(|(s, t)| (self.graph.node_data(s), self.graph.node_data(t)))
            .collect()
    }

    fn reachable_nodes(&self, node: &DepNode<K>, direction: Direction) -> Vec<&DepNode<K>> {
        if let Some(&index) = self.indices.get(node) {
            self.graph.depth_traverse(index, direction).map(|s| self.graph.node_data(s)).collect()
        } else {
            vec![]
        }
    }

    /// All nodes that can reach `node`.
    pub fn transitive_predecessors(&self, node: &DepNode<K>) -> Vec<&DepNode<K>> {
        self.reachable_nodes(node, INCOMING)
    }
}
