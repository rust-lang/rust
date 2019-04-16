use super::serialized::SerializedDepGraph;
use super::{DepKind, DepNode, DepNodeIndex, DepNodeState};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::AtomicCell;
use rustc_index::vec::IndexVec;

#[derive(Debug)]
pub struct PreviousDepGraph<K: DepKind> {
    pub(super) data: SerializedDepGraph<K>,
    pub(super) index: FxHashMap<DepNode<K>, DepNodeIndex>,
    pub(super) unused: Vec<DepNodeIndex>,
}

impl<K: DepKind> Default for PreviousDepGraph<K> {
    fn default() -> Self {
        Self { data: Default::default(), index: Default::default(), unused: Default::default() }
    }
}

impl<K: DepKind> PreviousDepGraph<K> {
    pub fn new_and_state(
        data: SerializedDepGraph<K>,
    ) -> (Self, IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>) {
        let mut unused = Vec::new();

        let state: IndexVec<_, _> = data
            .nodes
            .iter_enumerated()
            .map(|(index, node)| {
                if node.kind == DepKind::NULL {
                    // There might be `DepKind::NULL` nodes due to thread-local dep node indices
                    // that didn't get assigned anything.
                    // We also changed outdated nodes to `DepKind::NULL`.
                    unused.push(index);
                    AtomicCell::new(DepNodeState::Invalid)
                } else {
                    AtomicCell::new(DepNodeState::Unknown)
                }
            })
            .collect();

        let index: FxHashMap<_, _> =
            data.nodes
                .iter_enumerated()
                .filter_map(|(idx, &dep_node)| {
                    if dep_node.kind == DepKind::NULL { None } else { Some((dep_node, idx)) }
                })
                .collect();

        (PreviousDepGraph { data, index, unused }, state)
    }

    #[inline]
    pub fn edge_targets_from(&self, dep_node_index: DepNodeIndex) -> &[DepNodeIndex] {
        self.data.edge_targets_from(dep_node_index)
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: DepNodeIndex) -> DepNode<K> {
        self.data.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index(&self, dep_node: &DepNode<K>) -> DepNodeIndex {
        self.index[dep_node]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode<K>) -> Option<DepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode<K>) -> Option<Fingerprint> {
        self.index.get(dep_node).map(|&node_index| self.data.fingerprints[node_index])
    }

    #[inline]
    pub fn fingerprint_by_index(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        self.data.fingerprints[dep_node_index]
    }

    pub fn node_count(&self) -> usize {
        self.data.nodes.len()
    }
}
