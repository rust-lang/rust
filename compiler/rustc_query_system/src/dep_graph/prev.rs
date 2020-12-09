use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};
use super::{DepKind, DepNode};
use crate::dep_graph::DepContext;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::def_id::DefPathHash;

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
    pub fn index_to_node<CTX: DepContext<DepKind = K>>(
        &self,
        dep_node_index: SerializedDepNodeIndex,
        tcx: CTX,
    ) -> DepNode<K> {
        let dep_node = self.data.nodes[dep_node_index];
        // We have just loaded a deserialized `DepNode` from the previous
        // compilation session into the current one. If this was a foreign `DefId`,
        // then we stored additional information in the incr comp cache when we
        // initially created its fingerprint (see `DepNodeParams::to_fingerprint`)
        // We won't be calling `to_fingerprint` again for this `DepNode` (we no longer
        // have the original value), so we need to copy over this additional information
        // from the old incremental cache into the new cache that we serialize
        // and the end of this compilation session.
        if dep_node.kind.can_reconstruct_query_key() {
            tcx.register_reused_dep_path_hash(DefPathHash(dep_node.hash.into()));
        }
        dep_node
    }

    /// When debug assertions are enabled, asserts that the dep node at `dep_node_index` is equal to `dep_node`.
    /// This method should be preferred over manually calling `index_to_node`.
    /// Calls to `index_to_node` may affect global state, so gating a call
    /// to `index_to_node` on debug assertions could cause behavior changes when debug assertions
    /// are enabled.
    #[inline]
    pub fn debug_assert_eq(&self, dep_node_index: SerializedDepNodeIndex, dep_node: DepNode<K>) {
        debug_assert_eq!(self.data.nodes[dep_node_index], dep_node);
    }

    /// Obtains a debug-printable version of the `DepNode`.
    /// See `debug_assert_eq` for why this should be preferred over manually
    /// calling `dep_node_index`
    pub fn debug_dep_node(&self, dep_node_index: SerializedDepNodeIndex) -> impl std::fmt::Debug {
        // We're returning the `DepNode` without calling `register_reused_dep_path_hash`,
        // but `impl Debug` return type means that it can only be used for debug printing.
        // So, there's no risk of calls trying to create new dep nodes that have this
        // node as a dependency
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
