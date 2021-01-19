//! The data that we will serialize and deserialize.

use super::{DepKind, DepNode};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_index::vec::IndexVec;
use rustc_serialize::{Decodable, Decoder};

// The maximum value of `SerializedDepNodeIndex` leaves the upper two bits
// unused so that we can store multiple index types in `CompressedHybridIndex`,
// and use those bits to encode which index type it contains.
rustc_index::newtype_index! {
    pub struct SerializedDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug)]
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

impl<D: Decoder, K: DepKind + Decodable<D>> Decodable<D> for SerializedDepGraph<K> {
    fn decode(d: &mut D) -> Result<SerializedDepGraph<K>, D::Error> {
        // We used to serialize the dep graph by creating and serializing a `SerializedDepGraph`
        // using data copied from the `DepGraph`. But copying created a large memory spike, so we
        // now serialize directly from the `DepGraph` as if it's a `SerializedDepGraph`. Because we
        // deserialize that data into a `SerializedDepGraph` in the next compilation session, we
        // need `DepGraph`'s `Encodable` and `SerializedDepGraph`'s `Decodable` implementations to
        // be in sync. If you update this decoding, be sure to update the encoding, and vice-versa.
        //
        // We mimic the sequence of `Encode` and `Encodable` method calls used by the `DepGraph`'s
        // `Encodable` implementation with the corresponding sequence of `Decode` and `Decodable`
        // method calls. E.g. `Decode::read_struct` pairs with `Encode::emit_struct`, `DepNode`'s
        // `decode` pairs with `DepNode`'s `encode`, and so on. Any decoding methods not associated
        // with corresponding encoding methods called in `DepGraph`'s `Encodable` implementation
        // are off limits, because we'd be relying on their implementation details.
        //
        // For example, because we know it happens to do the right thing, its tempting to just use
        // `IndexVec`'s `Decodable` implementation to decode into some of the collections below,
        // even though `DepGraph` doesn't use its `Encodable` implementation. But the `IndexVec`
        // implementation could change, and we'd have a bug.
        //
        // Variables below are explicitly typed so that anyone who changes the `SerializedDepGraph`
        // representation without updating this function will encounter a compilation error, and
        // know to update this and possibly the `DepGraph` `Encodable` implementation accordingly
        // (the latter should serialize data in a format compatible with our representation).

        d.read_struct("SerializedDepGraph", 4, |d| {
            let nodes: IndexVec<SerializedDepNodeIndex, DepNode<K>> =
                d.read_struct_field("nodes", 0, |d| {
                    d.read_seq(|d, len| {
                        let mut v = IndexVec::with_capacity(len);
                        for i in 0..len {
                            v.push(d.read_seq_elt(i, |d| Decodable::decode(d))?);
                        }
                        Ok(v)
                    })
                })?;

            let fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint> =
                d.read_struct_field("fingerprints", 1, |d| {
                    d.read_seq(|d, len| {
                        let mut v = IndexVec::with_capacity(len);
                        for i in 0..len {
                            v.push(d.read_seq_elt(i, |d| Decodable::decode(d))?);
                        }
                        Ok(v)
                    })
                })?;

            let edge_list_indices: IndexVec<SerializedDepNodeIndex, (u32, u32)> = d
                .read_struct_field("edge_list_indices", 2, |d| {
                    d.read_seq(|d, len| {
                        let mut v = IndexVec::with_capacity(len);
                        for i in 0..len {
                            v.push(d.read_seq_elt(i, |d| Decodable::decode(d))?);
                        }
                        Ok(v)
                    })
                })?;

            let edge_list_data: Vec<SerializedDepNodeIndex> =
                d.read_struct_field("edge_list_data", 3, |d| {
                    d.read_seq(|d, len| {
                        let mut v = Vec::with_capacity(len);
                        for i in 0..len {
                            v.push(d.read_seq_elt(i, |d| Decodable::decode(d))?);
                        }
                        Ok(v)
                    })
                })?;

            Ok(SerializedDepGraph { nodes, fingerprints, edge_list_indices, edge_list_data })
        })
    }
}
