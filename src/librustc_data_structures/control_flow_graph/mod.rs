// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::indexed_vec::Idx;
pub use std::slice::Iter;

pub mod dominators;
pub mod iterate;
pub mod reachable;
mod reference;
pub mod transpose;

#[cfg(test)]
mod test;

pub trait ControlFlowGraph
    where Self: for<'graph> GraphPredecessors<'graph, Item=<Self as ControlFlowGraph>::Node>,
          Self: for<'graph> GraphSuccessors<'graph, Item=<Self as ControlFlowGraph>::Node>
{
    type Node: Idx;

    fn num_nodes(&self) -> usize;
    fn start_node(&self) -> Self::Node;
    fn predecessors<'graph>(&'graph self, node: Self::Node)
                            -> <Self as GraphPredecessors<'graph>>::Iter;
    fn successors<'graph>(&'graph self, node: Self::Node)
                            -> <Self as GraphSuccessors<'graph>>::Iter;
}

pub trait GraphPredecessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}

pub trait GraphSuccessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}
