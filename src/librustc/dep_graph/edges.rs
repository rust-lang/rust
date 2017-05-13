// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use std::fmt::Debug;
use std::hash::Hash;
use super::{DepGraphQuery, DepNode};

pub struct DepGraphEdges<D: Clone + Debug + Eq + Hash> {
    nodes: Vec<DepNode<D>>,
    indices: FxHashMap<DepNode<D>, IdIndex>,
    edges: FxHashSet<(IdIndex, IdIndex)>,
    open_nodes: Vec<OpenNode>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct IdIndex {
    index: u32
}

impl IdIndex {
    fn new(v: usize) -> IdIndex {
        assert!((v & 0xFFFF_FFFF) == v);
        IdIndex { index: v as u32 }
    }

    fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
enum OpenNode {
    Node(IdIndex),
    Ignore,
}

impl<D: Clone + Debug + Eq + Hash> DepGraphEdges<D> {
    pub fn new() -> DepGraphEdges<D> {
        DepGraphEdges {
            nodes: vec![],
            indices: FxHashMap(),
            edges: FxHashSet(),
            open_nodes: Vec::new()
        }
    }

    fn id(&self, index: IdIndex) -> DepNode<D> {
        self.nodes[index.index()].clone()
    }

    /// Creates a node for `id` in the graph.
    fn make_node(&mut self, id: DepNode<D>) -> IdIndex {
        if let Some(&i) = self.indices.get(&id) {
            return i;
        }

        let index = IdIndex::new(self.nodes.len());
        self.nodes.push(id.clone());
        self.indices.insert(id, index);
        index
    }

    /// Top of the stack of open nodes.
    fn current_node(&self) -> Option<OpenNode> {
        self.open_nodes.last().cloned()
    }

    pub fn push_ignore(&mut self) {
        self.open_nodes.push(OpenNode::Ignore);
    }

    pub fn pop_ignore(&mut self) {
        let popped_node = self.open_nodes.pop().unwrap();
        assert_eq!(popped_node, OpenNode::Ignore);
    }

    pub fn push_task(&mut self, key: DepNode<D>) {
        let top_node = self.current_node();

        let new_node = self.make_node(key);
        self.open_nodes.push(OpenNode::Node(new_node));

        // if we are in the midst of doing task T, then this new task
        // N is a subtask of T, so add an edge N -> T.
        if let Some(top_node) = top_node {
            self.add_edge_from_open_node(top_node, |t| (new_node, t));
        }
    }

    pub fn pop_task(&mut self, key: DepNode<D>) {
        let popped_node = self.open_nodes.pop().unwrap();
        assert_eq!(OpenNode::Node(self.indices[&key]), popped_node);
    }

    /// Indicates that the current task `C` reads `v` by adding an
    /// edge from `v` to `C`. If there is no current task, panics. If
    /// you want to suppress this edge, use `ignore`.
    pub fn read(&mut self, v: DepNode<D>) {
        let source = self.make_node(v);
        self.add_edge_from_current_node(|current| (source, current))
    }

    /// Indicates that the current task `C` writes `v` by adding an
    /// edge from `C` to `v`. If there is no current task, panics. If
    /// you want to suppress this edge, use `ignore`.
    pub fn write(&mut self, v: DepNode<D>) {
        let target = self.make_node(v);
        self.add_edge_from_current_node(|current| (current, target))
    }

    /// Invoke `add_edge_from_open_node` with the top of the stack, or
    /// panic if stack is empty.
    fn add_edge_from_current_node<OP>(&mut self,
                                      op: OP)
        where OP: FnOnce(IdIndex) -> (IdIndex, IdIndex)
    {
        match self.current_node() {
            Some(open_node) => self.add_edge_from_open_node(open_node, op),
            None => bug!("no current node, cannot add edge into dependency graph")
        }
    }

    /// Adds an edge to or from the `open_node`, assuming `open_node`
    /// is not `Ignore`. The direction of the edge is determined by
    /// the closure `op` --- we pass as argument the open node `n`,
    /// and the closure returns a (source, target) tuple, which should
    /// include `n` in one spot or another.
    fn add_edge_from_open_node<OP>(&mut self,
                                   open_node: OpenNode,
                                   op: OP)
        where OP: FnOnce(IdIndex) -> (IdIndex, IdIndex)
    {
        let (source, target) = match open_node {
            OpenNode::Node(n) => op(n),
            OpenNode::Ignore => { return; }
        };

        // ignore trivial self edges, which are not very interesting
        if source == target {
            return;
        }

        if self.edges.insert((source, target)) {
            debug!("adding edge from {:?} to {:?}",
                   self.id(source),
                   self.id(target));
        }
    }

    pub fn query(&self) -> DepGraphQuery<D> {
        let edges: Vec<_> = self.edges.iter()
                                      .map(|&(i, j)| (self.id(i), self.id(j)))
                                      .collect();
        DepGraphQuery::new(&self.nodes, &edges)
    }
}
