//! The data that we will serialize and deserialize.

use crate::dep_graph::DepNode;
use crate::ich::Fingerprint;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};

newtype_index! {
    pub struct SerializedDepNodeIndex { .. }
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug, RustcEncodable, RustcDecodable, Default)]
pub struct SerializedDepGraph {
    /// The set of all DepNodes in the graph
    pub nodes: IndexVec<SerializedDepNodeIndex, DepNode>,
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

impl SerializedDepGraph {
    #[inline]
    pub fn edge_targets_from(&self,
                             source: SerializedDepNodeIndex)
                             -> &[SerializedDepNodeIndex] {
        let targets = self.edge_list_indices[source];
        &self.edge_list_data[targets.0 as usize..targets.1 as usize]
    }
}
