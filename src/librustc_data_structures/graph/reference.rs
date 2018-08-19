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

impl<'graph, G: DirectedGraph> DirectedGraph for &'graph G {
    type Node = G::Node;
}

impl<'graph, G: WithNumNodes> WithNumNodes for &'graph G {
    fn num_nodes(&self) -> usize {
        (**self).num_nodes()
    }
}

impl<'graph, G: WithStartNode> WithStartNode for &'graph G {
    fn start_node(&self) -> Self::Node {
        (**self).start_node()
    }
}

impl<'graph, G: WithSuccessors> WithSuccessors for &'graph G {
    fn successors<'iter>(&'iter self, node: Self::Node) -> <Self as GraphSuccessors<'iter>>::Iter {
        (**self).successors(node)
    }
}

impl<'graph, G: WithPredecessors> WithPredecessors for &'graph G {
    fn predecessors<'iter>(&'iter self,
                           node: Self::Node)
                           -> <Self as GraphPredecessors<'iter>>::Iter {
        (**self).predecessors(node)
    }
}

impl<'iter, 'graph, G: WithPredecessors> GraphPredecessors<'iter> for &'graph G {
    type Item = G::Node;
    type Iter = <G as GraphPredecessors<'iter>>::Iter;
}

impl<'iter, 'graph, G: WithSuccessors> GraphSuccessors<'iter> for &'graph G {
    type Item = G::Node;
    type Iter = <G as GraphSuccessors<'iter>>::Iter;
}
