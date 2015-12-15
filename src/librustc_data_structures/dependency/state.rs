// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fnv::{FnvHashSet, FnvHashMap};
use graph::{Graph, NodeIndex};

use super::DepNodeId;

pub struct DepGraphState<ID: DepNodeId> {
    graph: Graph<ID, ()>,
    nodes: FnvHashMap<ID, NodeIndex>,
    edges: FnvHashSet<(NodeIndex, NodeIndex)>,
    open_nodes: Vec<OpenNode>,
}

#[derive(Clone, Debug, PartialEq)]
enum OpenNode {
    Node(NodeIndex),
    Ignore,
}

impl<ID: DepNodeId> DepGraphState<ID> {
    pub fn new() -> DepGraphState<ID> {
        DepGraphState {
            graph: Graph::new(),
            nodes: FnvHashMap(),
            edges: FnvHashSet(),
            open_nodes: Vec::new()
        }
    }

    /// Creates an entry for `node` in the graph.
    fn make_node(&mut self, node: ID) -> NodeIndex {
        let graph = &mut self.graph;
        *self.nodes.entry(node.clone())
                   .or_insert_with(|| graph.add_node(node))
    }

    /// Top of the stack of open nodes.
    fn current_node(&self) -> Option<OpenNode> {
        self.open_nodes.last().cloned()
    }

    /// All nodes reachable from `node`. In other words, things that
    /// will have to be recomputed if `node` changes.
    pub fn dependents(&self, node: ID) -> Vec<ID> {
        match self.nodes.get(&node) {
            None => vec![],
            Some(&index) =>
                self.graph.depth_traverse(index)
                          .map(|dependent_node| self.graph.node_data(dependent_node).clone())
                          .collect()
        }
    }

    pub fn push_ignore(&mut self) {
        self.open_nodes.push(OpenNode::Ignore);
    }

    pub fn pop_ignore(&mut self) {
        let popped_node = self.open_nodes.pop().unwrap();
        assert_eq!(popped_node, OpenNode::Ignore);
    }

    pub fn push_task(&mut self, key: ID) {
        let top_node = self.current_node();

        let new_node = self.make_node(key.clone());
        self.open_nodes.push(OpenNode::Node(new_node));

        // if we are in the midst of doing task T, then this new task
        // N is a subtask of T, so add an edge N -> T.
        if let Some(top_node) = top_node {
            self.add_edge_from_open_node(top_node, |t| (new_node, t));
        }
    }

    pub fn pop_task(&mut self, key: ID) {
        let popped_node = self.open_nodes.pop().unwrap();
        assert_eq!(OpenNode::Node(self.nodes[&key]), popped_node);
    }

    /// Indicates that the current task `C` reads `v` by adding an
    /// edge from `v` to `C`. If there is no current task, panics. If
    /// you want to suppress this edge, use `ignore`.
    pub fn read(&mut self, v: ID) {
        let source = self.make_node(v);
        self.add_edge_from_current_node(|current| (source, current))
    }

    /// Indicates that the current task `C` writes `v` by adding an
    /// edge from `C` to `v`. If there is no current task, panics. If
    /// you want to suppress this edge, use `ignore`.
    pub fn write(&mut self, v: ID) {
        let target = self.make_node(v);
        self.add_edge_from_current_node(|current| (current, target))
    }

    /// Invoke `add_edge_from_open_node` with the top of the stack, or
    /// panic if stack is empty.
    fn add_edge_from_current_node<OP>(&mut self,
                                      op: OP)
        where OP: FnOnce(NodeIndex) -> (NodeIndex, NodeIndex)
    {
        match self.current_node() {
            Some(open_node) => self.add_edge_from_open_node(open_node, op),
            None => panic!("no current node, cannot add edge into dependency graph")
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
        where OP: FnOnce(NodeIndex) -> (NodeIndex, NodeIndex)
    {
        let (source, target) = match open_node {
            OpenNode::Node(n) => op(n),
            OpenNode::Ignore => { return; }
        };

        if self.edges.insert((source, target)) {
            debug!("adding edge from {:?} to {:?}",
                   self.graph.node_data(source),
                   self.graph.node_data(target));
            self.graph.add_edge(source, target, ());
        }
    }

    pub fn nodes(&self) -> Vec<ID> {
        self.nodes.keys().cloned().collect()
    }

    pub fn edges(&self) -> Vec<(ID,ID)> {
        self.graph.all_edges()
                  .iter()
                  .map(|edge| (edge.source(), edge.target()))
                  .map(|(source, target)| (self.graph.node_data(source).clone(),
                                           self.graph.node_data(target).clone()))
                  .collect()
    }
}
