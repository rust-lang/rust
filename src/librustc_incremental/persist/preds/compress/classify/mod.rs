// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! First phase. Detect cycles and cross-edges.

use super::*;

#[cfg(test)]
mod test;

pub struct Classify<'a, 'g: 'a, N: 'g, I: 'a, O: 'a>
    where N: Debug + Clone + 'g,
          I: Fn(&N) -> bool,
          O: Fn(&N) -> bool,
{
    r: &'a mut GraphReduce<'g, N, I, O>,
    stack: Vec<NodeIndex>,
    colors: Vec<Color>,
    dag: Dag,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Color {
    // not yet visited
    White,

    // visiting; usize is index on stack
    Grey(usize),

    // finished visiting
    Black,
}

impl<'a, 'g, N, I, O> Classify<'a, 'g, N, I, O>
    where N: Debug + Clone + 'g,
          I: Fn(&N) -> bool,
          O: Fn(&N) -> bool,
{
    pub(super) fn new(r: &'a mut GraphReduce<'g, N, I, O>) -> Self {
        Classify {
            r: r,
            colors: vec![Color::White; r.in_graph.len_nodes()],
            stack: vec![],
            dag: Dag {
                parents: (0..r.in_graph.len_nodes()).map(|i| NodeIndex(i)).collect(),
                cross_edges: vec![],
                input_nodes: vec![],
                output_nodes: vec![],
            },
        }
    }

    pub(super) fn walk(mut self) -> Dag {
        for (index, node) in self.r.in_graph.all_nodes().iter().enumerate() {
            if (self.r.is_output)(&node.data) {
                let index = NodeIndex(index);
                self.dag.output_nodes.push(index);
                match self.colors[index.0] {
                    Color::White => self.open(index),
                    Color::Grey(_) => panic!("grey node but have not yet started a walk"),
                    Color::Black => (), // already visited, skip
                }
            }
        }

        // At this point we've identifed all the cycles, and we've
        // constructed a spanning tree over the original graph
        // (encoded in `self.parents`) as well as a list of
        // cross-edges that reflect additional edges from the DAG.
        //
        // If we converted each node to its `cycle-head` (a
        // representative choice from each SCC, basically) and then
        // take the union of `self.parents` and `self.cross_edges`
        // (after canonicalization), that is basically our DAG.
        //
        // Note that both of those may well contain trivial `X -rf-> X`
        // cycle edges after canonicalization, though. e.g., if you
        // have a graph `{A -rf-> B, B -rf-> A}`, we will have unioned A and
        // B, but A will also be B's parent (or vice versa), and hence
        // when we canonicalize the parent edge it would become `A -rf->
        // A` (or `B -rf-> B`).
        self.dag
    }

    fn open(&mut self, node: NodeIndex) {
        let index = self.stack.len();
        self.stack.push(node);
        self.colors[node.0] = Color::Grey(index);
        for child in self.r.inputs(node) {
            self.walk_edge(node, child);
        }
        self.stack.pop().unwrap();
        self.colors[node.0] = Color::Black;

        if (self.r.is_input)(&self.r.in_graph.node_data(node)) {
            // base inputs should have no inputs
            assert!(self.r.inputs(node).next().is_none());
            debug!("input: `{:?}`", self.r.in_graph.node_data(node));
            self.dag.input_nodes.push(node);
        }
    }

    fn walk_edge(&mut self, parent: NodeIndex, child: NodeIndex) {
        debug!("walk_edge: {:?} -rf-> {:?}, {:?}",
               self.r.in_graph.node_data(parent),
               self.r.in_graph.node_data(child),
               self.colors[child.0]);

        // Ignore self-edges, just in case they exist.
        if child == parent {
            return;
        }

        match self.colors[child.0] {
            Color::White => {
                // Not yet visited this node; start walking it.
                assert_eq!(self.dag.parents[child.0], child);
                self.dag.parents[child.0] = parent;
                self.open(child);
            }

            Color::Grey(stack_index) => {
                // Back-edge; unify everything on stack between here and `stack_index`
                // since we are all participating in a cycle
                assert!(self.stack[stack_index] == child);

                for &n in &self.stack[stack_index..] {
                    debug!("cycle `{:?}` and `{:?}`",
                           self.r.in_graph.node_data(n),
                           self.r.in_graph.node_data(parent));
                    self.r.mark_cycle(n, parent);
                }
            }

            Color::Black => {
                // Cross-edge, record and ignore
                self.dag.cross_edges.push((parent, child));
                debug!("cross-edge `{:?} -rf-> {:?}`",
                       self.r.in_graph.node_data(parent),
                       self.r.in_graph.node_data(child));
            }
        }
    }
}
