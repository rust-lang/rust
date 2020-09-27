use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};
use super::{DepKind, DepNode};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;

#[derive(Debug, Encodable, Decodable)]
pub struct PreviousDepGraph<K: DepKind> {
    data: SerializedDepGraph<K>,
    index: FxHashMap<DepNode<K>, SerializedDepNodeIndex>,
}

impl<K: DepKind> Default for PreviousDepGraph<K> {
    fn default() -> Self {
        PreviousDepGraph { data: Default::default(), index: Default::default() }
    }
}

impl<K: DepKind> PreviousDepGraph<K> {
    pub fn new(data: SerializedDepGraph<K>) -> PreviousDepGraph<K> {
        let index: FxHashMap<_, _> =
            data.nodes.iter_enumerated().map(|(idx, &dep_node)| (dep_node, idx)).collect();
        PreviousDepGraph { data, index }
    }

    #[inline]
    pub fn edge_targets_from(
        &self,
        dep_node_index: SerializedDepNodeIndex,
    ) -> &[SerializedDepNodeIndex] {
        self.data.edge_targets_from(dep_node_index)
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode<K> {
        self.data.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index(&self, dep_node: &DepNode<K>) -> SerializedDepNodeIndex {
        self.index[dep_node]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode<K>) -> Option<SerializedDepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode<K>) -> Option<Fingerprint> {
        self.index.get(dep_node).map(|&node_index| self.data.fingerprints[node_index])
    }

    #[inline]
    pub fn fingerprint_by_index(&self, dep_node_index: SerializedDepNodeIndex) -> Fingerprint {
        self.data.fingerprints[dep_node_index]
    }

    pub fn node_count(&self) -> usize {
        self.index.len()
    }
}
