// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A graph module for use in dataflow, region resolution, and elsewhere.
//!
//! # Interface details
//!
//! You customize the graph by specifying a "node data" type `N` and an
//! "edge data" type `E`. You can then later gain access (mutable or
//! immutable) to these "user-data" bits. Currently, you can only add
//! nodes or edges to the graph. You cannot remove or modify them once
//! added. This could be changed if we have a need.
//!
//! # Implementation details
//!
//! The main tricky thing about this code is the way that edges are
//! stored. The edges are stored in a central array, but they are also
//! threaded onto two linked lists for each node, one for incoming edges
//! and one for outgoing edges. Note that every edge is a member of some
//! incoming list and some outgoing list.  Basically you can load the
//! first index of the linked list from the node data structures (the
//! field `first_edge`) and then, for each edge, load the next index from
//! the field `next_edge`). Each of those fields is an array that should
//! be indexed by the direction (see the type `Direction`).

use bitvec::BitVector;
use std::fmt::{Formatter, Error, Debug};
use std::usize;
use snapshot_vec::{SnapshotVec, SnapshotVecDelegate};

#[cfg(test)]
mod tests;

pub struct Graph<N, E> {
    nodes: SnapshotVec<Node<N>>,
    edges: SnapshotVec<Edge<E>>,
}

pub struct Node<N> {
    first_edge: [EdgeIndex; 2], // see module comment
    pub data: N,
}

pub struct Edge<E> {
    next_edge: [EdgeIndex; 2], // see module comment
    source: NodeIndex,
    target: NodeIndex,
    pub data: E,
}

impl<N> SnapshotVecDelegate for Node<N> {
    type Value = Node<N>;
    type Undo = ();

    fn reverse(_: &mut Vec<Node<N>>, _: ()) {}
}

impl<N> SnapshotVecDelegate for Edge<N> {
    type Value = Edge<N>;
    type Undo = ();

    fn reverse(_: &mut Vec<Edge<N>>, _: ()) {}
}

impl<E: Debug> Debug for Edge<E> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f,
               "Edge {{ next_edge: [{:?}, {:?}], source: {:?}, target: {:?}, data: {:?} }}",
               self.next_edge[0],
               self.next_edge[1],
               self.source,
               self.target,
               self.data)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NodeIndex(pub usize);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct EdgeIndex(pub usize);

pub const INVALID_EDGE_INDEX: EdgeIndex = EdgeIndex(usize::MAX);

// Use a private field here to guarantee no more instances are created:
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Direction {
    repr: usize,
}

pub const OUTGOING: Direction = Direction { repr: 0 };

pub const INCOMING: Direction = Direction { repr: 1 };

impl NodeIndex {
    /// Returns unique id (unique with respect to the graph holding associated node).
    pub fn node_id(&self) -> usize {
        self.0
    }
}

impl EdgeIndex {
    /// Returns unique id (unique with respect to the graph holding associated edge).
    pub fn edge_id(&self) -> usize {
        self.0
    }
}

impl<N: Debug, E: Debug> Graph<N, E> {
    pub fn new() -> Graph<N, E> {
        Graph {
            nodes: SnapshotVec::new(),
            edges: SnapshotVec::new(),
        }
    }

    // # Simple accessors

    #[inline]
    pub fn all_nodes(&self) -> &[Node<N>] {
        &self.nodes
    }

    #[inline]
    pub fn len_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn all_edges(&self) -> &[Edge<E>] {
        &self.edges
    }

    #[inline]
    pub fn len_edges(&self) -> usize {
        self.edges.len()
    }

    // # Node construction

    pub fn next_node_index(&self) -> NodeIndex {
        NodeIndex(self.nodes.len())
    }

    pub fn add_node(&mut self, data: N) -> NodeIndex {
        let idx = self.next_node_index();
        self.nodes.push(Node {
            first_edge: [INVALID_EDGE_INDEX, INVALID_EDGE_INDEX],
            data: data,
        });
        idx
    }

    pub fn mut_node_data(&mut self, idx: NodeIndex) -> &mut N {
        &mut self.nodes[idx.0].data
    }

    pub fn node_data(&self, idx: NodeIndex) -> &N {
        &self.nodes[idx.0].data
    }

    pub fn node(&self, idx: NodeIndex) -> &Node<N> {
        &self.nodes[idx.0]
    }

    // # Edge construction and queries

    pub fn next_edge_index(&self) -> EdgeIndex {
        EdgeIndex(self.edges.len())
    }

    pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex, data: E) -> EdgeIndex {
        debug!("graph: add_edge({:?}, {:?}, {:?})", source, target, data);

        let idx = self.next_edge_index();

        // read current first of the list of edges from each node
        let source_first = self.nodes[source.0].first_edge[OUTGOING.repr];
        let target_first = self.nodes[target.0].first_edge[INCOMING.repr];

        // create the new edge, with the previous firsts from each node
        // as the next pointers
        self.edges.push(Edge {
            next_edge: [source_first, target_first],
            source: source,
            target: target,
            data: data,
        });

        // adjust the firsts for each node target be the next object.
        self.nodes[source.0].first_edge[OUTGOING.repr] = idx;
        self.nodes[target.0].first_edge[INCOMING.repr] = idx;

        return idx;
    }

    pub fn mut_edge_data(&mut self, idx: EdgeIndex) -> &mut E {
        &mut self.edges[idx.0].data
    }

    pub fn edge_data(&self, idx: EdgeIndex) -> &E {
        &self.edges[idx.0].data
    }

    pub fn edge(&self, idx: EdgeIndex) -> &Edge<E> {
        &self.edges[idx.0]
    }

    pub fn first_adjacent(&self, node: NodeIndex, dir: Direction) -> EdgeIndex {
        //! Accesses the index of the first edge adjacent to `node`.
        //! This is useful if you wish to modify the graph while walking
        //! the linked list of edges.

        self.nodes[node.0].first_edge[dir.repr]
    }

    pub fn next_adjacent(&self, edge: EdgeIndex, dir: Direction) -> EdgeIndex {
        //! Accesses the next edge in a given direction.
        //! This is useful if you wish to modify the graph while walking
        //! the linked list of edges.

        self.edges[edge.0].next_edge[dir.repr]
    }

    // # Iterating over nodes, edges

    pub fn enumerated_nodes(&self) -> EnumeratedNodes<N> {
        EnumeratedNodes {
            iter: self.nodes.iter().enumerate()
        }
    }

    pub fn enumerated_edges(&self) -> EnumeratedEdges<E> {
        EnumeratedEdges {
            iter: self.edges.iter().enumerate()
        }
    }

    pub fn each_node<'a, F>(&'a self, mut f: F) -> bool
        where F: FnMut(NodeIndex, &'a Node<N>) -> bool
    {
        //! Iterates over all edges defined in the graph.
        self.enumerated_nodes().all(|(node_idx, node)| f(node_idx, node))
    }

    pub fn each_edge<'a, F>(&'a self, mut f: F) -> bool
        where F: FnMut(EdgeIndex, &'a Edge<E>) -> bool
    {
        //! Iterates over all edges defined in the graph
        self.enumerated_edges().all(|(edge_idx, edge)| f(edge_idx, edge))
    }

    pub fn outgoing_edges(&self, source: NodeIndex) -> AdjacentEdges<N, E> {
        self.adjacent_edges(source, OUTGOING)
    }

    pub fn incoming_edges(&self, source: NodeIndex) -> AdjacentEdges<N, E> {
        self.adjacent_edges(source, INCOMING)
    }

    pub fn adjacent_edges(&self, source: NodeIndex, direction: Direction) -> AdjacentEdges<N, E> {
        let first_edge = self.node(source).first_edge[direction.repr];
        AdjacentEdges {
            graph: self,
            direction: direction,
            next: first_edge,
        }
    }

    pub fn successor_nodes(&self, source: NodeIndex) -> AdjacentTargets<N, E> {
        self.outgoing_edges(source).targets()
    }

    pub fn predecessor_nodes(&self, target: NodeIndex) -> AdjacentSources<N, E> {
        self.incoming_edges(target).sources()
    }

    /// A common use for graphs in our compiler is to perform
    /// fixed-point iteration. In this case, each edge represents a
    /// constraint, and the nodes themselves are associated with
    /// variables or other bitsets. This method facilitates such a
    /// computation.
    pub fn iterate_until_fixed_point<'a, F>(&'a self, mut op: F)
        where F: FnMut(usize, EdgeIndex, &'a Edge<E>) -> bool
    {
        let mut iteration = 0;
        let mut changed = true;
        while changed {
            changed = false;
            iteration += 1;
            for (edge_index, edge) in self.enumerated_edges() {
                changed |= op(iteration, edge_index, edge);
            }
        }
    }

    pub fn depth_traverse<'a>(&'a self,
                              start: NodeIndex,
                              direction: Direction)
                              -> DepthFirstTraversal<'a, N, E> {
        DepthFirstTraversal::with_start_node(self, start, direction)
    }

    /// Whether or not a node can be reached from itself.
    pub fn is_node_cyclic(&self, starting_node_index: NodeIndex) -> bool {
        // This is similar to depth traversal below, but we
        // can't use that, because depth traversal doesn't show
        // the starting node a second time.
        let mut visited = BitVector::new(self.len_nodes());
        let mut stack = vec![starting_node_index];

        while let Some(current_node_index) = stack.pop() {
            visited.insert(current_node_index.0);

            // Directionality doesn't change the answer,
            // so just use outgoing edges.
            for (_, edge) in self.outgoing_edges(current_node_index) {
                let target_node_index = edge.target();

                if target_node_index == starting_node_index {
                    return true;
                }

                if !visited.contains(target_node_index.0) {
                    stack.push(target_node_index);
                }
            }
        }

        false
    }
}

// # Iterators

pub struct EnumeratedNodes<'g, N>
    where N: 'g,
{
    iter: ::std::iter::Enumerate<::std::slice::Iter<'g, Node<N>>>
}

impl<'g, N: Debug> Iterator for EnumeratedNodes<'g, N> {
    type Item = (NodeIndex, &'g Node<N>);

    fn next(&mut self) -> Option<(NodeIndex, &'g Node<N>)> {
        self.iter.next().map(|(idx, n)| (NodeIndex(idx), n))
    }
}

pub struct EnumeratedEdges<'g, E>
    where E: 'g,
{
    iter: ::std::iter::Enumerate<::std::slice::Iter<'g, Edge<E>>>
}

impl<'g, E: Debug> Iterator for EnumeratedEdges<'g, E> {
    type Item = (EdgeIndex, &'g Edge<E>);

    fn next(&mut self) -> Option<(EdgeIndex, &'g Edge<E>)> {
        self.iter.next().map(|(idx, e)| (EdgeIndex(idx), e))
    }
}

pub struct AdjacentEdges<'g, N, E>
    where N: 'g,
          E: 'g
{
    graph: &'g Graph<N, E>,
    direction: Direction,
    next: EdgeIndex,
}

impl<'g, N, E> AdjacentEdges<'g, N, E> {
    fn targets(self) -> AdjacentTargets<'g, N, E> {
        AdjacentTargets { edges: self }
    }

    fn sources(self) -> AdjacentSources<'g, N, E> {
        AdjacentSources { edges: self }
    }
}

impl<'g, N: Debug, E: Debug> Iterator for AdjacentEdges<'g, N, E> {
    type Item = (EdgeIndex, &'g Edge<E>);

    fn next(&mut self) -> Option<(EdgeIndex, &'g Edge<E>)> {
        let edge_index = self.next;
        if edge_index == INVALID_EDGE_INDEX {
            return None;
        }

        let edge = self.graph.edge(edge_index);
        self.next = edge.next_edge[self.direction.repr];
        Some((edge_index, edge))
    }
}

pub struct AdjacentTargets<'g, N, E>
    where N: 'g,
          E: 'g
{
    edges: AdjacentEdges<'g, N, E>,
}

impl<'g, N: Debug, E: Debug> Iterator for AdjacentTargets<'g, N, E> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<NodeIndex> {
        self.edges.next().map(|(_, edge)| edge.target)
    }
}

pub struct AdjacentSources<'g, N, E>
    where N: 'g,
          E: 'g
{
    edges: AdjacentEdges<'g, N, E>,
}

impl<'g, N: Debug, E: Debug> Iterator for AdjacentSources<'g, N, E> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<NodeIndex> {
        self.edges.next().map(|(_, edge)| edge.source)
    }
}

pub struct DepthFirstTraversal<'g, N, E>
    where N: 'g,
          E: 'g
{
    graph: &'g Graph<N, E>,
    stack: Vec<NodeIndex>,
    visited: BitVector,
    direction: Direction,
}

impl<'g, N: Debug, E: Debug> DepthFirstTraversal<'g, N, E> {
    pub fn new(graph: &'g Graph<N, E>, direction: Direction) -> Self {
        let visited = BitVector::new(graph.len_nodes());
        DepthFirstTraversal {
            graph: graph,
            stack: vec![],
            visited: visited,
            direction: direction,
        }
    }

    pub fn with_start_node(graph: &'g Graph<N, E>,
                           start_node: NodeIndex,
                           direction: Direction)
                           -> Self {
        let mut visited = BitVector::new(graph.len_nodes());
        visited.insert(start_node.node_id());
        DepthFirstTraversal {
            graph: graph,
            stack: vec![start_node],
            visited: visited,
            direction: direction,
        }
    }

    pub fn reset(&mut self, start_node: NodeIndex) {
        self.stack.truncate(0);
        self.stack.push(start_node);
        self.visited.clear();
        self.visited.insert(start_node.node_id());
    }

    fn visit(&mut self, node: NodeIndex) {
        if self.visited.insert(node.node_id()) {
            self.stack.push(node);
        }
    }
}

impl<'g, N: Debug, E: Debug> Iterator for DepthFirstTraversal<'g, N, E> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<NodeIndex> {
        let next = self.stack.pop();
        if let Some(idx) = next {
            for (_, edge) in self.graph.adjacent_edges(idx, self.direction) {
                let target = edge.source_or_target(self.direction);
                self.visit(target);
            }
        }
        next
    }
}

pub fn each_edge_index<F>(max_edge_index: EdgeIndex, mut f: F)
    where F: FnMut(EdgeIndex) -> bool
{
    let mut i = 0;
    let n = max_edge_index.0;
    while i < n {
        if !f(EdgeIndex(i)) {
            return;
        }
        i += 1;
    }
}

impl<E> Edge<E> {
    pub fn source(&self) -> NodeIndex {
        self.source
    }

    pub fn target(&self) -> NodeIndex {
        self.target
    }

    pub fn source_or_target(&self, direction: Direction) -> NodeIndex {
        if direction == OUTGOING {
            self.target
        } else {
            self.source
        }
    }
}
