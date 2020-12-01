use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::implementation::{Direction, Graph, NodeIndex, INCOMING};

use super::{DepKind, DepNode};

pub struct DepGraphQuery<K> {
    pub graph: Graph<DepNode<K>, ()>,
    pub indices: FxHashMap<DepNode<K>, NodeIndex>,
}

impl<K: DepKind> DepGraphQuery<K> {
    pub fn new(
        nodes: &[DepNode<K>],
        edge_list_indices: &[(usize, usize)],
        edge_list_data: &[usize],
    ) -> DepGraphQuery<K> {
        let mut graph = Graph::with_capacity(nodes.len(), edge_list_data.len());
        let mut indices = FxHashMap::default();
        for node in nodes {
            indices.insert(*node, graph.add_node(*node));
        }

        for (source, &(start, end)) in edge_list_indices.iter().enumerate() {
            for &target in &edge_list_data[start..end] {
                let source = indices[&nodes[source]];
                let target = indices[&nodes[target]];
                graph.add_edge(source, target, ());
            }
        }

        DepGraphQuery { graph, indices }
    }

    pub fn contains_node(&self, node: &DepNode<K>) -> bool {
        self.indices.contains_key(&node)
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
