// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ich::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use super::dep_node::DepNode;
use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct PreviousDepGraph {
    data: SerializedDepGraph,
    index: FxHashMap<DepNode, SerializedDepNodeIndex>,
}

impl PreviousDepGraph {
    pub fn new(data: SerializedDepGraph) -> PreviousDepGraph {
        let index: FxHashMap<_, _> = data.nodes
            .iter_enumerated()
            .map(|(idx, &(dep_node, _))| (dep_node, idx))
            .collect();
        PreviousDepGraph { data, index }
    }

    pub fn with_edges_from<F>(&self, dep_node: &DepNode, mut f: F)
    where
        F: FnMut(&(DepNode, Fingerprint)),
    {
        let node_index = self.index[dep_node];
        self.data
            .edge_targets_from(node_index)
            .into_iter()
            .for_each(|&index| f(&self.data.nodes[index]));
    }

    pub fn fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.index
            .get(dep_node)
            .map(|&node_index| self.data.nodes[node_index].1)
    }
}
