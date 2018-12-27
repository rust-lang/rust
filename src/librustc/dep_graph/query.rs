use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::implementation::{
    Direction, INCOMING, Graph, NodeIndex, OUTGOING
};

use super::DepNode;

pub struct DepGraphQuery {
    pub graph: Graph<DepNode, ()>,
    pub indices: FxHashMap<DepNode, NodeIndex>,
}

impl DepGraphQuery {
    pub fn new(nodes: &[DepNode],
               edges: &[(DepNode, DepNode)])
               -> DepGraphQuery {
        let mut graph = Graph::with_capacity(nodes.len(), edges.len());
        let mut indices = FxHashMap::default();
        for node in nodes {
            indices.insert(node.clone(), graph.add_node(node.clone()));
        }

        for &(ref source, ref target) in edges {
            let source = indices[source];
            let target = indices[target];
            graph.add_edge(source, target, ());
        }

        DepGraphQuery {
            graph,
            indices,
        }
    }

    pub fn contains_node(&self, node: &DepNode) -> bool {
        self.indices.contains_key(&node)
    }

    pub fn nodes(&self) -> Vec<&DepNode> {
        self.graph.all_nodes()
                  .iter()
                  .map(|n| &n.data)
                  .collect()
    }

    pub fn edges(&self) -> Vec<(&DepNode,&DepNode)> {
        self.graph.all_edges()
                  .iter()
                  .map(|edge| (edge.source(), edge.target()))
                  .map(|(s, t)| (self.graph.node_data(s),
                                 self.graph.node_data(t)))
                  .collect()
    }

    fn reachable_nodes(&self, node: &DepNode, direction: Direction) -> Vec<&DepNode> {
        if let Some(&index) = self.indices.get(node) {
            self.graph.depth_traverse(index, direction)
                      .map(|s| self.graph.node_data(s))
                      .collect()
        } else {
            vec![]
        }
    }

    /// All nodes reachable from `node`. In other words, things that
    /// will have to be recomputed if `node` changes.
    pub fn transitive_successors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.reachable_nodes(node, OUTGOING)
    }

    /// All nodes that can reach `node`.
    pub fn transitive_predecessors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.reachable_nodes(node, INCOMING)
    }

    /// Just the outgoing edges from `node`.
    pub fn immediate_successors(&self, node: &DepNode) -> Vec<&DepNode> {
        if let Some(&index) = self.indices.get(&node) {
            self.graph.successor_nodes(index)
                      .map(|s| self.graph.node_data(s))
                      .collect()
        } else {
            vec![]
        }
    }
}
