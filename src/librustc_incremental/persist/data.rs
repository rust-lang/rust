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
use rustc_serialize::{Decoder as RustcDecoder, Encoder as RustcEncoder};

use super::directory::DefPathIndex;

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedDepGraph {
    pub nodes: Vec<DepNode<DefPathIndex>>,
    pub edges: Vec<SerializedEdge>,
    pub hashes: Vec<SerializedHash>,
}

pub type SerializedEdge = (DepNode<DefPathIndex>, DepNode<DefPathIndex>);

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedHash {
    pub index: DefPathIndex,

    /// the hash itself, computed by `calculate_item_hash`
    pub hash: u64,
}
