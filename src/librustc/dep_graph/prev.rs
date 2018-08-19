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
            .map(|(idx, &dep_node)| (dep_node, idx))
            .collect();
        PreviousDepGraph { data, index }
    }

    #[inline]
    pub fn edges_from(&self,
                      dep_node: &DepNode)
                      -> Option<(&[SerializedDepNodeIndex], SerializedDepNodeIndex)> {
        self.index
            .get(dep_node)
            .map(|&node_index| {
                (self.data.edge_targets_from(node_index), node_index)
            })
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode {
        self.data.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index(&self, dep_node: &DepNode) -> SerializedDepNodeIndex {
        self.index[dep_node]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode) -> Option<SerializedDepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.index
            .get(dep_node)
            .map(|&node_index| self.data.fingerprints[node_index])
    }

    #[inline]
    pub fn fingerprint_by_index(&self,
                                dep_node_index: SerializedDepNodeIndex)
                                -> Fingerprint {
        self.data.fingerprints[dep_node_index]
    }

    pub fn node_count(&self) -> usize {
        self.index.len()
    }
}
