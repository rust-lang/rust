// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;

pub struct TransposedGraph<G: ControlFlowGraph> {
    base_graph: G,
    start_node: G::Node,
}

impl<G: ControlFlowGraph> TransposedGraph<G> {
    pub fn new(base_graph: G) -> Self {
        let start_node = base_graph.start_node();
        Self::with_start(base_graph, start_node)
    }

    pub fn with_start(base_graph: G, start_node: G::Node) -> Self {
        TransposedGraph {
            base_graph: base_graph,
            start_node: start_node,
        }
    }
}

impl<G: ControlFlowGraph> ControlFlowGraph for TransposedGraph<G> {
    type Node = G::Node;

    fn num_nodes(&self) -> usize {
        self.base_graph.num_nodes()
    }

    fn start_node(&self) -> Self::Node {
        self.start_node
    }

    fn predecessors<'graph>(&'graph self,
                            node: Self::Node)
                            -> <Self as GraphPredecessors<'graph>>::Iter {
        self.base_graph.successors(node)
    }

    fn successors<'graph>(&'graph self,
                          node: Self::Node)
                          -> <Self as GraphSuccessors<'graph>>::Iter {
        self.base_graph.predecessors(node)
    }
}

impl<'graph, G: ControlFlowGraph> GraphPredecessors<'graph> for TransposedGraph<G> {
    type Item = G::Node;
    type Iter = <G as GraphSuccessors<'graph>>::Iter;
}

impl<'graph, G: ControlFlowGraph> GraphSuccessors<'graph> for TransposedGraph<G> {
    type Item = G::Node;
    type Iter = <G as GraphPredecessors<'graph>>::Iter;
}
