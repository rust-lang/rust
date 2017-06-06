// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The data that we will serialize and deserialize.

use rustc::dep_graph::{DepNode, WorkProduct, WorkProductId};
use rustc::hir::def_id::DefIndex;
use rustc::hir::map::DefPathHash;
use rustc::ich::Fingerprint;
use rustc::middle::cstore::EncodedMetadataHash;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};

/// Data for use when recompiling the **current crate**.
#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedDepGraph {
    /// The set of all DepNodes in the graph
    pub nodes: IndexVec<DepNodeIndex, DepNode<DefPathHash>>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    pub edge_list_indices: Vec<(u32, u32)>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    pub edge_list_data: Vec<DepNodeIndex>,

    /// These are output nodes that have no incoming edges. We track
    /// these separately so that when we reload all edges, we don't
    /// lose track of these nodes.
    pub bootstrap_outputs: Vec<DepNode<DefPathHash>>,

    /// These are hashes of two things:
    /// - the HIR nodes in this crate
    /// - the metadata nodes from dependent crates we use
    ///
    /// In each case, we store a hash summarizing the contents of
    /// those items as they were at the time we did this compilation.
    /// In the case of HIR nodes, this hash is derived by walking the
    /// HIR itself. In the case of metadata nodes, the hash is loaded
    /// from saved state.
    ///
    /// When we do the next compile, we will load these back up and
    /// compare them against the hashes we see at that time, which
    /// will tell us what has changed, either in this crate or in some
    /// crate that we depend on.
    ///
    /// Because they will be reloaded, we don't store the DefId (which
    /// will be different when we next compile) related to each node,
    /// but rather the `DefPathIndex`. This can then be retraced
    /// to find the current def-id.
    pub hashes: Vec<SerializedHash>,
}

/// The index of a DepNode in the SerializedDepGraph::nodes array.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug,
         RustcEncodable, RustcDecodable)]
pub struct DepNodeIndex(pub u32);

impl DepNodeIndex {
    #[inline]
    pub fn new(idx: usize) -> DepNodeIndex {
        assert!(idx <= ::std::u32::MAX as usize);
        DepNodeIndex(idx as u32)
    }
}

impl Idx for DepNodeIndex {
    #[inline]
    fn new(idx: usize) -> Self {
        assert!(idx <= ::std::u32::MAX as usize);
        DepNodeIndex(idx as u32)
    }

    #[inline]
    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedHash {
    /// def-id of thing being hashed
    pub dep_node: DepNode<DefPathHash>,

    /// the hash as of previous compilation, computed by code in
    /// `hash` module
    pub hash: Fingerprint,
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedWorkProduct {
    /// node that produced the work-product
    pub id: WorkProductId,

    /// work-product data itself
    pub work_product: WorkProduct,
}

/// Data for use when downstream crates get recompiled.
#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedMetadataHashes {
    /// For each def-id defined in this crate that appears in the
    /// metadata, we hash all the inputs that were used when producing
    /// the metadata. We save this after compilation is done. Then,
    /// when some downstream crate is being recompiled, it can compare
    /// the hashes we saved against the hashes that it saw from
    /// before; this will tell it which of the items in this crate
    /// changed, which in turn implies what items in the downstream
    /// crate need to be recompiled.
    ///
    /// Note that we store the def-ids here. This is because we don't
    /// reload this file when we recompile this crate, we will just
    /// regenerate it completely with the current hashes and new def-ids.
    ///
    /// Then downstream creates will load up their
    /// `SerializedDepGraph`, which may contain `MetaData(X)` nodes
    /// where `X` refers to some item in this crate. That `X` will be
    /// a `DefPathIndex` that gets retracted to the current `DefId`
    /// (matching the one found in this structure).
    pub entry_hashes: Vec<EncodedMetadataHash>,

    /// This map contains fingerprints that are not specific to some DefId but
    /// describe something global to the whole crate.
    pub global_hashes: Vec<(DepNode<()>, Fingerprint)>,

    /// For each DefIndex (as it occurs in SerializedMetadataHash), this
    /// map stores the DefPathIndex (as it occurs in DefIdDirectory), so
    /// that we can find the new DefId for a SerializedMetadataHash in a
    /// subsequent compilation session.
    ///
    /// This map is only needed for running auto-tests using the
    /// #[rustc_metadata_dirty] and #[rustc_metadata_clean] attributes, and
    /// is only populated if -Z query-dep-graph is specified. It will be
    /// empty otherwise. Importing crates are perfectly happy with just having
    /// the DefIndex.
    pub index_map: FxHashMap<DefIndex, DefPathHash>
}
