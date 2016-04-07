// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::fnv::FnvHashMap;
use rustc_data_structures::graph::{Graph, NodeIndex};
use std::fmt::Debug;
use std::hash::Hash;

use super::DepNode;

pub struct DepGraphQuery<D: Clone + Debug + Hash + Eq> {
    pub graph: Graph<DepNode<D>, ()>,
    pub indices: FnvHashMap<DepNode<D>, NodeIndex>,
}

impl<D: Clone + Debug + Hash + Eq> DepGraphQuery<D> {
    pub fn new(nodes: &[DepNode<D>],
               edges: &[(DepNode<D>, DepNode<D>)])
               -> DepGraphQuery<D> {
        let mut graph = Graph::new();
        let mut indices = FnvHashMap();
        for node in nodes {
            indices.insert(node.clone(), graph.next_node_index());
            graph.add_node(node.clone());
        }

        for &(ref source, ref target) in edges {
            let source = indices[source];
            let target = indices[target];
            graph.add_edge(source, target, ());
        }

        DepGraphQuery {
            graph: graph,
            indices: indices
        }
    }

    pub fn contains_node(&self, node: &DepNode<D>) -> bool {
        self.indices.contains_key(&node)
    }

    pub fn nodes(&self) -> Vec<DepNode<D>> {
        self.graph.all_nodes()
                  .iter()
                  .map(|n| n.data.clone())
                  .collect()
    }

    pub fn edges(&self) -> Vec<(DepNode<D>,DepNode<D>)> {
        self.graph.all_edges()
                  .iter()
                  .map(|edge| (edge.source(), edge.target()))
                  .map(|(s, t)| (self.graph.node_data(s).clone(),
                                 self.graph.node_data(t).clone()))
                  .collect()
    }

    /// All nodes reachable from `node`. In other words, things that
    /// will have to be recomputed if `node` changes.
    pub fn transitive_dependents(&self, node: DepNode<D>) -> Vec<DepNode<D>> {
        if let Some(&index) = self.indices.get(&node) {
            self.graph.depth_traverse(index)
                      .map(|s| self.graph.node_data(s).clone())
                      .collect()
        } else {
            vec![]
        }
    }

    /// Just the outgoing edges from `node`.
    pub fn immediate_dependents(&self, node: DepNode<D>) -> Vec<DepNode<D>> {
        if let Some(&index) = self.indices.get(&node) {
            self.graph.successor_nodes(index)
                      .map(|s| self.graph.node_data(s).clone())
                      .collect()
        } else {
            vec![]
        }
    }
}
