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
//! incoming list and some outgoing list. Basically you can load the
//! first index of the linked list from the node data structures (the
//! field `first_edge`) and then, for each edge, load the next index from
//! the field `next_edge`). Each of those fields is an array that should
//! be indexed by the direction (see the type `Direction`).

use crate::snapshot_vec::{SnapshotVec, SnapshotVecDelegate};
use rustc_index::bit_set::BitSet;
use std::fmt::Debug;

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

#[derive(Debug)]
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

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NodeIndex(pub usize);

#[derive(Copy, Clone, PartialEq, Debug)]
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
    /// Returns unique ID (unique with respect to the graph holding associated node).
    pub fn node_id(self) -> usize {
        self.0
    }
}

impl<N: Debug, E: Debug> Graph<N, E> {
    pub fn new() -> Graph<N, E> {
        Graph { nodes: SnapshotVec::new(), edges: SnapshotVec::new() }
    }

    pub fn with_capacity(nodes: usize, edges: usize) -> Graph<N, E> {
        Graph { nodes: SnapshotVec::with_capacity(nodes), edges: SnapshotVec::with_capacity(edges) }
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
        self.nodes.push(Node { first_edge: [INVALID_EDGE_INDEX, INVALID_EDGE_INDEX], data });
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
        self.edges.push(Edge { next_edge: [source_first, target_first], source, target, data });

        // adjust the firsts for each node target be the next object.
        self.nodes[source.0].first_edge[OUTGOING.repr] = idx;
        self.nodes[target.0].first_edge[INCOMING.repr] = idx;

        idx
    }

    pub fn edge(&self, idx: EdgeIndex) -> &Edge<E> {
        &self.edges[idx.0]
    }

    // # Iterating over nodes, edges

    pub fn enumerated_nodes(&self) -> impl Iterator<Item = (NodeIndex, &Node<N>)> {
        self.nodes.iter().enumerate().map(|(idx, n)| (NodeIndex(idx), n))
    }

    pub fn enumerated_edges(&self) -> impl Iterator<Item = (EdgeIndex, &Edge<E>)> {
        self.edges.iter().enumerate().map(|(idx, e)| (EdgeIndex(idx), e))
    }

    pub fn each_node<'a>(&'a self, mut f: impl FnMut(NodeIndex, &'a Node<N>) -> bool) -> bool {
        //! Iterates over all edges defined in the graph.
        self.enumerated_nodes().all(|(node_idx, node)| f(node_idx, node))
    }

    pub fn each_edge<'a>(&'a self, mut f: impl FnMut(EdgeIndex, &'a Edge<E>) -> bool) -> bool {
        //! Iterates over all edges defined in the graph
        self.enumerated_edges().all(|(edge_idx, edge)| f(edge_idx, edge))
    }

    pub fn outgoing_edges(&self, source: NodeIndex) -> AdjacentEdges<'_, N, E> {
        self.adjacent_edges(source, OUTGOING)
    }

    pub fn incoming_edges(&self, source: NodeIndex) -> AdjacentEdges<'_, N, E> {
        self.adjacent_edges(source, INCOMING)
    }

    pub fn adjacent_edges(
        &self,
        source: NodeIndex,
        direction: Direction,
    ) -> AdjacentEdges<'_, N, E> {
        let first_edge = self.node(source).first_edge[direction.repr];
        AdjacentEdges { graph: self, direction, next: first_edge }
    }

    pub fn successor_nodes<'a>(
        &'a self,
        source: NodeIndex,
    ) -> impl Iterator<Item = NodeIndex> + 'a {
        self.outgoing_edges(source).targets()
    }

    pub fn predecessor_nodes<'a>(
        &'a self,
        target: NodeIndex,
    ) -> impl Iterator<Item = NodeIndex> + 'a {
        self.incoming_edges(target).sources()
    }

    pub fn depth_traverse(
        &self,
        start: NodeIndex,
        direction: Direction,
    ) -> DepthFirstTraversal<'_, N, E> {
        DepthFirstTraversal::with_start_node(self, start, direction)
    }

    pub fn nodes_in_postorder(
        &self,
        direction: Direction,
        entry_node: NodeIndex,
    ) -> Vec<NodeIndex> {
        let mut visited = BitSet::new_empty(self.len_nodes());
        let mut stack = vec![];
        let mut result = Vec::with_capacity(self.len_nodes());
        let mut push_node = |stack: &mut Vec<_>, node: NodeIndex| {
            if visited.insert(node.0) {
                stack.push((node, self.adjacent_edges(node, direction)));
            }
        };

        for node in
            Some(entry_node).into_iter().chain(self.enumerated_nodes().map(|(node, _)| node))
        {
            push_node(&mut stack, node);
            while let Some((node, mut iter)) = stack.pop() {
                if let Some((_, child)) = iter.next() {
                    let target = child.source_or_target(direction);
                    // the current node needs more processing, so
                    // add it back to the stack
                    stack.push((node, iter));
                    // and then push the new node
                    push_node(&mut stack, target);
                } else {
                    result.push(node);
                }
            }
        }

        assert_eq!(result.len(), self.len_nodes());
        result
    }
}

// # Iterators

pub struct AdjacentEdges<'g, N, E> {
    graph: &'g Graph<N, E>,
    direction: Direction,
    next: EdgeIndex,
}

impl<'g, N: Debug, E: Debug> AdjacentEdges<'g, N, E> {
    fn targets(self) -> impl Iterator<Item = NodeIndex> + 'g {
        self.map(|(_, edge)| edge.target)
    }

    fn sources(self) -> impl Iterator<Item = NodeIndex> + 'g {
        self.map(|(_, edge)| edge.source)
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        // At most, all the edges in the graph.
        (0, Some(self.graph.len_edges()))
    }
}

pub struct DepthFirstTraversal<'g, N, E> {
    graph: &'g Graph<N, E>,
    stack: Vec<NodeIndex>,
    visited: BitSet<usize>,
    direction: Direction,
}

impl<'g, N: Debug, E: Debug> DepthFirstTraversal<'g, N, E> {
    pub fn with_start_node(
        graph: &'g Graph<N, E>,
        start_node: NodeIndex,
        direction: Direction,
    ) -> Self {
        let mut visited = BitSet::new_empty(graph.len_nodes());
        visited.insert(start_node.node_id());
        DepthFirstTraversal { graph, stack: vec![start_node], visited, direction }
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We will visit every node in the graph exactly once.
        let remaining = self.graph.len_nodes() - self.visited.count();
        (remaining, Some(remaining))
    }
}

impl<'g, N: Debug, E: Debug> ExactSizeIterator for DepthFirstTraversal<'g, N, E> {}

impl<E> Edge<E> {
    pub fn source(&self) -> NodeIndex {
        self.source
    }

    pub fn target(&self) -> NodeIndex {
        self.target
    }

    pub fn source_or_target(&self, direction: Direction) -> NodeIndex {
        if direction == OUTGOING { self.target } else { self.source }
    }
}
