// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Graph compression. See `README.md`.

use rustc_data_structures::graph::{Graph, NodeIndex};
use rustc_data_structures::unify::UnificationTable;
use std::fmt::Debug;

#[cfg(test)]
#[macro_use]
mod test_macro;

mod construct;

mod classify;
use self::classify::Classify;

mod dag_id;
use self::dag_id::DagId;

#[cfg(test)]
mod test;

pub fn reduce_graph<N, I, O>(graph: &Graph<N, ()>,
                             is_input: I,
                             is_output: O) -> Reduction<N>
    where N: Debug + Clone,
          I: Fn(&N) -> bool,
          O: Fn(&N) -> bool,
{
    GraphReduce::new(graph, is_input, is_output).compute()
}

pub struct Reduction<'q, N> where N: 'q + Debug + Clone {
    pub graph: Graph<&'q N, ()>,
    pub input_nodes: Vec<NodeIndex>,
}

struct GraphReduce<'q, N, I, O>
    where N: 'q + Debug + Clone,
          I: Fn(&N) -> bool,
          O: Fn(&N) -> bool,
{
    in_graph: &'q Graph<N, ()>,
    unify: UnificationTable<DagId>,
    is_input: I,
    is_output: O,
}

struct Dag {
    // The "parent" of a node is the node which reached it during the
    // initial DFS. To encode the case of "no parent" (i.e., for the
    // roots of the walk), we make `parents[i] == i` to start, which
    // turns out be convenient.
    parents: Vec<NodeIndex>,

    // Additional edges beyond the parents.
    cross_edges: Vec<(NodeIndex, NodeIndex)>,

    // Nodes which we found that are considered "outputs"
    output_nodes: Vec<NodeIndex>,

    // Nodes which we found that are considered "inputs"
    input_nodes: Vec<NodeIndex>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct DagNode {
    in_index: NodeIndex
}

impl<'q, N, I, O> GraphReduce<'q, N, I, O>
    where N: Debug + Clone,
          I: Fn(&N) -> bool,
          O: Fn(&N) -> bool,
{
    fn new(in_graph: &'q Graph<N, ()>, is_input: I, is_output: O) -> Self {
        let mut unify = UnificationTable::new();

        // create a set of unification keys whose indices
        // correspond to the indices from the input graph
        for i in 0..in_graph.len_nodes() {
            let k = unify.new_key(());
            assert!(k == DagId::from_input_index(NodeIndex(i)));
        }

        GraphReduce { in_graph, unify, is_input, is_output }
    }

    fn compute(mut self) -> Reduction<'q, N> {
        let dag = Classify::new(&mut self).walk();
        construct::construct_graph(&mut self, dag)
    }

    fn inputs(&self, in_node: NodeIndex) -> impl Iterator<Item = NodeIndex> + 'q {
        self.in_graph.predecessor_nodes(in_node)
    }

    fn mark_cycle(&mut self, in_node1: NodeIndex, in_node2: NodeIndex) {
        let dag_id1 = DagId::from_input_index(in_node1);
        let dag_id2 = DagId::from_input_index(in_node2);
        self.unify.union(dag_id1, dag_id2);
    }

    /// Convert a dag-id into its cycle head representative. This will
    /// be a no-op unless `in_node` participates in a cycle, in which
    /// case a distinct node *may* be returned.
    fn cycle_head(&mut self, in_node: NodeIndex) -> NodeIndex {
        let i = DagId::from_input_index(in_node);
        self.unify.find(i).as_input_index()
    }

    #[cfg(test)]
    fn in_cycle(&mut self, ni1: NodeIndex, ni2: NodeIndex) -> bool {
        self.cycle_head(ni1) == self.cycle_head(ni2)
    }
}
