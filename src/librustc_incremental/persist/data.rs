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

use rustc::dep_graph::DepNode;

use super::directory::DefPathIndex;

/// Data for use when recompiling the **current crate**.
#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedDepGraph {
    pub nodes: Vec<DepNode<DefPathIndex>>,
    pub edges: Vec<SerializedEdge>,

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
    pub hashes: Vec<SerializedHash>,
}

/// Data for use when downstream crates get recompiled.
#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedMetadataHashes {
    /// For each def-id defined in this crate that appears in the
    /// metadata, we hash all the inputs that were used when producing
    /// the metadata. We save this after compilation is done.  Then,
    /// when some downstream crate is being recompiled, it can compare
    /// the hashes we saved against the hashes that it saw from
    /// before; this will tell it which of the items in this crate
    /// changed, which in turn implies what items in the downstream
    /// crate need to be recompiled.
    pub hashes: Vec<SerializedHash>,
}

pub type SerializedEdge = (DepNode<DefPathIndex>, DepNode<DefPathIndex>);

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedHash {
    /// node being hashed; either a Hir or MetaData variant, in
    /// practice
    pub node: DepNode<DefPathIndex>,

    /// the hash itself, computed by `calculate_item_hash`
    pub hash: u64,
}
