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

use super::DepNode;

pub struct DepGraphQuery {
    pub graph: Graph<DepNode, ()>,
    pub indices: FnvHashMap<DepNode, NodeIndex>,
}

impl DepGraphQuery {
    pub fn new(nodes: &[DepNode], edges: &[(DepNode, DepNode)]) -> DepGraphQuery {
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

    pub fn nodes(&self) -> Vec<DepNode> {
        self.graph.all_nodes()
                  .iter()
                  .map(|n| n.data.clone())
                  .collect()
    }

    pub fn edges(&self) -> Vec<(DepNode,DepNode)> {
        self.graph.all_edges()
                  .iter()
                  .map(|edge| (edge.source(), edge.target()))
                  .map(|(s, t)| (self.graph.node_data(s).clone(), self.graph.node_data(t).clone()))
                  .collect()
    }

    /// All nodes reachable from `node`. In other words, things that
    /// will have to be recomputed if `node` changes.
    pub fn dependents(&self, node: DepNode) -> Vec<DepNode> {
        if let Some(&index) = self.indices.get(&node) {
            self.graph.depth_traverse(index)
                      .map(|dependent_node| self.graph.node_data(dependent_node).clone())
                      .collect()
        } else {
            vec![]
        }
    }
}
