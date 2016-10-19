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

impl<'graph, G: ControlFlowGraph> ControlFlowGraph for &'graph G {
    type Node = G::Node;

    fn num_nodes(&self) -> usize {
        (**self).num_nodes()
    }

    fn start_node(&self) -> Self::Node {
        (**self).start_node()
    }

    fn predecessors<'iter>(&'iter self,
                           node: Self::Node)
                           -> <Self as GraphPredecessors<'iter>>::Iter {
        (**self).predecessors(node)
    }

    fn successors<'iter>(&'iter self, node: Self::Node) -> <Self as GraphSuccessors<'iter>>::Iter {
        (**self).successors(node)
    }
}

impl<'iter, 'graph, G: ControlFlowGraph> GraphPredecessors<'iter> for &'graph G {
    type Item = G::Node;
    type Iter = <G as GraphPredecessors<'iter>>::Iter;
}

impl<'iter, 'graph, G: ControlFlowGraph> GraphSuccessors<'iter> for &'graph G {
    type Item = G::Node;
    type Iter = <G as GraphSuccessors<'iter>>::Iter;
}
