// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Compute reachability using a simple dataflow propagation.
//! Store end-result in a big NxN bit matrix.

use super::ControlFlowGraph;
use super::super::bitvec::BitVector;
use super::iterate::reverse_post_order;
use super::super::indexed_vec::{IndexVec, Idx};

#[cfg(test)]
mod test;

pub fn reachable<G: ControlFlowGraph>(graph: &G) -> Reachability<G::Node> {
    let reverse_post_order = reverse_post_order(graph, graph.start_node());
    reachable_given_rpo(graph, &reverse_post_order)
}

pub fn reachable_given_rpo<G: ControlFlowGraph>(graph: &G,
                                                reverse_post_order: &[G::Node])
                                                -> Reachability<G::Node> {
    let mut reachability = Reachability::new(graph);
    let mut changed = true;
    while changed {
        changed = false;
        for &node in reverse_post_order.iter().rev() {
            // every node can reach itself
            changed |= reachability.bits[node].insert(node.index());

            // and every pred can reach everything node can reach
            for pred in graph.predecessors(node) {
                let nodes_bits = reachability.bits[node].clone();
                changed |= reachability.bits[pred].insert_all(&nodes_bits);
            }
        }
    }
    reachability
}

pub struct Reachability<Node: Idx> {
    bits: IndexVec<Node, BitVector>,
}

impl<Node: Idx> Reachability<Node> {
    fn new<G: ControlFlowGraph>(graph: &G) -> Self {
        let num_nodes = graph.num_nodes();
        Reachability { bits: IndexVec::from_elem_n(BitVector::new(num_nodes), num_nodes) }
    }

    pub fn can_reach(&self, source: Node, target: Node) -> bool {
        let bit: usize = target.index();
        self.bits[source].contains(bit)
    }
}
