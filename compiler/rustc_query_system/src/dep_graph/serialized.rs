//! The data that we will serialize and deserialize.

use super::{DepKind, DepNode};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_index::vec::IndexVec;

// The maximum value of `SerializedDepNodeIndex` leaves the upper two bits
// unused so that we can store multiple index types in `CompressedHybridIndex`,
// and use those bits to encode which index type it contains.
rustc_index::newtype_index! {
    pub struct SerializedDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug, Encodable, Decodable)]
pub struct SerializedDepGraph<K: DepKind> {
    /// The set of all DepNodes in the graph
    pub nodes: IndexVec<SerializedDepNodeIndex, DepNode<K>>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    pub fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    pub edge_list_indices: IndexVec<SerializedDepNodeIndex, (u32, u32)>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    pub edge_list_data: Vec<SerializedDepNodeIndex>,
}

impl<K: DepKind> Default for SerializedDepGraph<K> {
    fn default() -> Self {
        SerializedDepGraph {
            nodes: Default::default(),
            fingerprints: Default::default(),
            edge_list_indices: Default::default(),
            edge_list_data: Default::default(),
        }
    }
}

impl<K: DepKind> SerializedDepGraph<K> {
    #[inline]
    pub fn edge_targets_from(&self, source: SerializedDepNodeIndex) -> &[SerializedDepNodeIndex] {
        let targets = self.edge_list_indices[source];
        &self.edge_list_data[targets.0 as usize..targets.1 as usize]
    }
}
