use crate::ich::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use super::dep_node::DepNode;
use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};

#[derive(Debug, RustcEncodable, RustcDecodable, Default)]
pub struct PreviousDepGraph {
    data: SerializedDepGraph,
    index: FxHashMap<DepNode, SerializedDepNodeIndex>,
}

impl PreviousDepGraph {
    pub fn new(data: SerializedDepGraph) -> PreviousDepGraph {
        let index: FxHashMap<_, _> = data.nodes
            .iter_enumerated()
            .map(|(idx, &dep_node)| (dep_node, idx))
            .collect();
        PreviousDepGraph { data, index }
    }

    #[inline]
    pub fn edge_targets_from(
        &self,
        dep_node_index: SerializedDepNodeIndex
    ) -> &[SerializedDepNodeIndex] {
        self.data.edge_targets_from(dep_node_index)
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode {
        self.data.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index(&self, dep_node: &DepNode) -> SerializedDepNodeIndex {
        self.index[dep_node]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode) -> Option<SerializedDepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.index
            .get(dep_node)
            .map(|&node_index| self.data.fingerprints[node_index])
    }

    #[inline]
    pub fn fingerprint_by_index(&self,
                                dep_node_index: SerializedDepNodeIndex)
                                -> Fingerprint {
        self.data.fingerprints[dep_node_index]
    }

    pub fn node_count(&self) -> usize {
        self.index.len()
    }
}
