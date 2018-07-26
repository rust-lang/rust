// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The data that we will serialize and deserialize.

use dep_graph::DepNode;
use ich::Fingerprint;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};

newtype_index!(SerializedDepNodeIndex);

/// Data for use when recompiling the **current crate**.
#[derive(Debug, RustcEncodable, RustcDecodable)]
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

    pub fn new() -> SerializedDepGraph {
        SerializedDepGraph {
            nodes: IndexVec::new(),
            fingerprints: IndexVec::new(),
            edge_list_indices: IndexVec::new(),
            edge_list_data: Vec::new(),
        }
    }

    #[inline]
    pub fn edge_targets_from(&self,
                             source: SerializedDepNodeIndex)
                             -> &[SerializedDepNodeIndex] {
        let targets = self.edge_list_indices[source];
        &self.edge_list_data[targets.0 as usize..targets.1 as usize]
    }
}
