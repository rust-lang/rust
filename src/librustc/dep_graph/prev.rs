use crate::ich::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::IndexVec;
use super::dep_node::DepNode;
use super::graph::DepNodeIndex;

#[derive(Debug, Default)]
pub struct PreviousDepGraph {
    /// Maps from dep nodes to their previous index, if any.
    pub(super) index: FxHashMap<DepNode, DepNodeIndex>,
    /// The set of all DepNodes in the graph
    pub(super) nodes: IndexVec<DepNodeIndex, DepNode>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    pub(super) fingerprints: IndexVec<DepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode.
    pub(super) edges: IndexVec<DepNodeIndex, Option<Box<[DepNodeIndex]>>>,
}

impl PreviousDepGraph {
    #[inline]
    pub fn edge_targets_from(
        &self,
        dep_node_index: DepNodeIndex
    ) -> &[DepNodeIndex] {
        self.edges[dep_node_index].as_ref().unwrap()
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: DepNodeIndex) -> DepNode {
        self.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index(&self, dep_node: &DepNode) -> DepNodeIndex {
        self.index[dep_node]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode) -> Option<DepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.index
            .get(dep_node)
            .map(|&node_index| self.fingerprints[node_index])
    }

    #[inline]
    pub fn fingerprint_by_index(&self,
                                dep_node_index: DepNodeIndex)
                                -> Fingerprint {
        self.fingerprints[dep_node_index]
    }

    pub fn node_count(&self) -> usize {
        self.index.len()
    }
}
