// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::dep_graph::{DepGraphQuery, DepNode};
use rustc::hir::def_id::DefId;
use rustc_data_structures::fnv::FnvHashMap;
use rustc_data_structures::graph::{DepthFirstTraversal, INCOMING, NodeIndex};

use super::hash::*;

/// A data-structure that makes it easy to enumerate the hashable
/// predecessors of any given dep-node.
pub struct Predecessors<'query> {
    // - Keys: dep-nodes that may have work-products, output meta-data
    //   nodes.
    // - Values: transitive predecessors of the key that are hashable
    //   (e.g., HIR nodes, input meta-data nodes)
    pub inputs: FnvHashMap<&'query DepNode<DefId>, Vec<&'query DepNode<DefId>>>,

    // - Keys: some hashable node
    // - Values: the hash thereof
    pub hashes: FnvHashMap<&'query DepNode<DefId>, u64>,
}

impl<'q> Predecessors<'q> {
    pub fn new(query: &'q DepGraphQuery<DefId>, hcx: &mut HashContext) -> Self {
        // Find nodes for which we want to know the full set of preds
        let mut dfs = DepthFirstTraversal::new(&query.graph, INCOMING);
        let all_nodes = query.graph.all_nodes();
        let tcx = hcx.tcx;

        let inputs: FnvHashMap<_, _> = all_nodes.iter()
            .enumerate()
            .filter(|&(_, node)| match node.data {
                DepNode::WorkProduct(_) => true,
                DepNode::MetaData(ref def_id) => def_id.is_local(),

                // if -Z query-dep-graph is passed, save more extended data
                // to enable better unit testing
                DepNode::TypeckItemBody(_) |
                DepNode::TransCrateItem(_) => tcx.sess.opts.debugging_opts.query_dep_graph,

                _ => false,
            })
            .map(|(node_index, node)| {
                dfs.reset(NodeIndex(node_index));
                let inputs: Vec<_> = dfs.by_ref()
                    .map(|i| &all_nodes[i.node_id()].data)
                    .filter(|d| HashContext::is_hashable(d))
                    .collect();
                (&node.data, inputs)
            })
            .collect();

        let mut hashes = FnvHashMap();
        for input in inputs.values().flat_map(|v| v.iter().cloned()) {
            hashes.entry(input)
                  .or_insert_with(|| hcx.hash(input).unwrap());
        }

        Predecessors {
            inputs: inputs,
            hashes: hashes,
        }
    }
}
