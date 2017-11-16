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

use rustc::dep_graph::{WorkProduct, WorkProductId};
use rustc::hir::map::DefPathHash;
use rustc::middle::cstore::EncodedMetadataHash;
use rustc_data_structures::fx::FxHashMap;

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
    pub index_map: FxHashMap<u32, DefPathHash>
}
