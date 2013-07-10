// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

A graph module for use in dataflow, region resolution, and elsewhere.

# Interface details

You customize the graph by specifying a "node data" type `N` and an
"edge data" type `E`. You can then later gain access (mutable or
immutable) to these "user-data" bits. Currently, you can only add
nodes or edges to the graph. You cannot remove or modify them once
added. This could be changed if we have a need.

# Implementation details

The main tricky thing about this code is the way that edges are
stored. The edges are stored in a central array, but they are also
threaded onto two linked lists for each node, one for incoming edges
and one for outgoing edges. Note that every edge is a member of some
incoming list and some outgoing list.  Basically you can load the
first index of the linked list from the node data structures (the
field `first_edge`) and then, for each edge, load the next index from
the field `next_edge`). Each of those fields is an array that should
be indexed by the direction (see the type `Direction`).

*/

use std::uint;
use std::vec;

pub struct Graph<N,E> {
    priv nodes: ~[Node<N>],
    priv edges: ~[Edge<E>],
}

pub struct Node<N> {
    priv first_edge: [EdgeIndex, ..2], // see module comment
    data: N,
}

pub struct Edge<E> {
    priv next_edge: [EdgeIndex, ..2], // see module comment
    priv source: NodeIndex,
    priv target: NodeIndex,
    data: E,
}

#[deriving(Eq)]
pub struct NodeIndex(uint);
pub static InvalidNodeIndex: NodeIndex = NodeIndex(uint::max_value);

#[deriving(Eq)]
pub struct EdgeIndex(uint);
pub static InvalidEdgeIndex: EdgeIndex = EdgeIndex(uint::max_value);

// Use a private field here to guarantee no more instances are created:
pub struct Direction { priv repr: uint }
pub static Outgoing: Direction = Direction { repr: 0 };
pub static Incoming: Direction = Direction { repr: 1 };

impl<N,E> Graph<N,E> {
    pub fn new() -> Graph<N,E> {
        Graph {nodes: ~[], edges: ~[]}
    }

    pub fn with_capacity(num_nodes: uint,
                         num_edges: uint) -> Graph<N,E> {
        Graph {nodes: vec::with_capacity(num_nodes),
               edges: vec::with_capacity(num_edges)}
    }

    ///////////////////////////////////////////////////////////////////////////
    // Simple accessors

    #[inline]
    pub fn all_nodes<'a>(&'a self) -> &'a [Node<N>] {
        let nodes: &'a [Node<N>] = self.nodes;
        nodes
    }

    #[inline]
    pub fn all_edges<'a>(&'a self) -> &'a [Edge<E>] {
        let edges: &'a [Edge<E>] = self.edges;
        edges
    }

    ///////////////////////////////////////////////////////////////////////////
    // Node construction

    pub fn next_node_index(&self) -> NodeIndex {
        NodeIndex(self.nodes.len())
    }

    pub fn add_node(&mut self, data: N) -> NodeIndex {
        let idx = self.next_node_index();
        self.nodes.push(Node {
            first_edge: [InvalidEdgeIndex, InvalidEdgeIndex],
            data: data
        });
        idx
    }

    pub fn mut_node_data<'a>(&'a mut self, idx: NodeIndex) -> &'a mut N {
        &mut self.nodes[*idx].data
    }

    pub fn node_data<'a>(&'a self, idx: NodeIndex) -> &'a N {
        &self.nodes[*idx].data
    }

    pub fn node<'a>(&'a self, idx: NodeIndex) -> &'a Node<N> {
        &self.nodes[*idx]
    }

    ///////////////////////////////////////////////////////////////////////////
    // Edge construction and queries

    pub fn next_edge_index(&self) -> EdgeIndex {
        EdgeIndex(self.edges.len())
    }

    pub fn add_edge(&mut self,
                    source: NodeIndex,
                    target: NodeIndex,
                    data: E) -> EdgeIndex {
        let idx = self.next_edge_index();

        // read current first of the list of edges from each node
        let source_first = self.nodes[*source].first_edge[Outgoing.repr];
        let target_first = self.nodes[*target].first_edge[Incoming.repr];

        // create the new edge, with the previous firsts from each node
        // as the next pointers
        self.edges.push(Edge {
            next_edge: [source_first, target_first],
            source: source,
            target: target,
            data: data
        });

        // adjust the firsts for each node target be the next object.
        self.nodes[*source].first_edge[Outgoing.repr] = idx;
        self.nodes[*target].first_edge[Incoming.repr] = idx;

        return idx;
    }

    pub fn mut_edge_data<'a>(&'a mut self, idx: EdgeIndex) -> &'a mut E {
        &mut self.edges[*idx].data
    }

    pub fn edge_data<'a>(&'a self, idx: EdgeIndex) -> &'a E {
        &self.edges[*idx].data
    }

    pub fn edge<'a>(&'a self, idx: EdgeIndex) -> &'a Edge<E> {
        &self.edges[*idx]
    }

    pub fn first_adjacent(&self, node: NodeIndex, dir: Direction) -> EdgeIndex {
        //! Accesses the index of the first edge adjacent to `node`.
        //! This is useful if you wish to modify the graph while walking
        //! the linked list of edges.

        self.nodes[*node].first_edge[dir.repr]
    }

    pub fn next_adjacent(&self, edge: EdgeIndex, dir: Direction) -> EdgeIndex {
        //! Accesses the next edge in a given direction.
        //! This is useful if you wish to modify the graph while walking
        //! the linked list of edges.

        self.edges[*edge].next_edge[dir.repr]
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterating over nodes, edges

    pub fn each_node(&self, f: &fn(NodeIndex, &Node<N>) -> bool) -> bool {
        //! Iterates over all edges defined in the graph.

        uint::range(0, self.nodes.len(),
                    |i| f(NodeIndex(i), &self.nodes[i]))
    }

    pub fn each_edge(&self, f: &fn(EdgeIndex, &Edge<E>) -> bool) -> bool {
        //! Iterates over all edges defined in the graph.

        uint::range(0, self.nodes.len(),
                    |i| f(EdgeIndex(i), &self.edges[i]))
    }

    pub fn each_outgoing_edge(&self,
                              source: NodeIndex,
                              f: &fn(EdgeIndex, &Edge<E>) -> bool) -> bool {
        //! Iterates over all outgoing edges from the node `from`

        self.each_adjacent_edge(source, Outgoing, f)
    }

    pub fn each_incoming_edge(&self,
                              target: NodeIndex,
                              f: &fn(EdgeIndex, &Edge<E>) -> bool) -> bool {
        //! Iterates over all incoming edges to the node `target`

        self.each_adjacent_edge(target, Incoming, f)
    }

    pub fn each_adjacent_edge(&self,
                              node: NodeIndex,
                              dir: Direction,
                              f: &fn(EdgeIndex, &Edge<E>) -> bool) -> bool {
        //! Iterates over all edges adjacent to the node `node`
        //! in the direction `dir` (either `Outgoing` or `Incoming)

        let mut edge_idx = self.first_adjacent(node, dir);
        while edge_idx != InvalidEdgeIndex {
            let edge = &self.edges[*edge_idx];
            if !f(edge_idx, edge) {
                return false;
            }
            edge_idx = edge.next_edge[dir.repr];
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Fixed-point iteration
    //
    // A common use for graphs in our compiler is to perform
    // fixed-point iteration. In this case, each edge represents a
    // constaint, and the nodes themselves are associated with
    // variables or other bitsets. This method facilitates such a
    // computation.

    pub fn iterate_until_fixed_point(&self,
                                     op: &fn(iter_index: uint,
                                             edge_index: EdgeIndex,
                                             edge: &Edge<E>) -> bool) {
        let mut iteration = 0;
        let mut changed = true;
        while changed {
            changed = false;
            iteration += 1;
            for self.edges.iter().enumerate().advance |(i, edge)| {
                changed |= op(iteration, EdgeIndex(i), edge);
            }
        }
    }
}

pub fn each_edge_index(max_edge_index: EdgeIndex, f: &fn(EdgeIndex) -> bool) {
    let mut i = 0;
    let n = *max_edge_index;
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
}

#[cfg(test)]
mod test {
    use middle::graph::*;

    type TestNode = Node<&'static str>;
    type TestEdge = Edge<&'static str>;
    type TestGraph = Graph<&'static str, &'static str>;

    fn create_graph() -> TestGraph {
        let mut graph = Graph::new();

        // Create a simple graph
        //
        //    A -+> B --> C
        //       |  |     ^
        //       |  v     |
        //       F  D --> E

        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        let e = graph.add_node("E");
        let f = graph.add_node("F");

        graph.add_edge(a, b, "AB");
        graph.add_edge(b, c, "BC");
        graph.add_edge(b, d, "BD");
        graph.add_edge(d, e, "DE");
        graph.add_edge(e, c, "EC");
        graph.add_edge(f, b, "FB");

        return graph;
    }

    #[test]
    fn each_node() {
        let graph = create_graph();
        let expected = ["A", "B", "C", "D", "E", "F"];
        for graph.each_node |idx, node| {
            assert_eq!(&expected[*idx], graph.node_data(idx));
            assert_eq!(expected[*idx], node.data);
        }
    }

    #[test]
    fn each_edge() {
        let graph = create_graph();
        let expected = ["AB", "BC", "BD", "DE", "EC", "FB"];
        for graph.each_edge |idx, edge| {
            assert_eq!(&expected[*idx], graph.edge_data(idx));
            assert_eq!(expected[*idx], edge.data);
        }
    }

    fn test_adjacent_edges<N:Eq,E:Eq>(graph: &Graph<N,E>,
                                      start_index: NodeIndex,
                                      start_data: N,
                                      expected_incoming: &[(E,N)],
                                      expected_outgoing: &[(E,N)]) {
        assert_eq!(graph.node_data(start_index), &start_data);

        let mut counter = 0;
        for graph.each_incoming_edge(start_index) |edge_index, edge| {
            assert_eq!(graph.edge_data(edge_index), &edge.data);
            assert!(counter < expected_incoming.len());
            debug!("counter=%? expected=%? edge_index=%? edge=%?",
                   counter, expected_incoming[counter], edge_index, edge);
            match expected_incoming[counter] {
                (ref e, ref n) => {
                    assert_eq!(e, &edge.data);
                    assert_eq!(n, graph.node_data(edge.source));
                    assert_eq!(start_index, edge.target);
                }
            }
            counter += 1;
        }
        assert_eq!(counter, expected_incoming.len());

        let mut counter = 0;
        for graph.each_outgoing_edge(start_index) |edge_index, edge| {
            assert_eq!(graph.edge_data(edge_index), &edge.data);
            assert!(counter < expected_outgoing.len());
            debug!("counter=%? expected=%? edge_index=%? edge=%?",
                   counter, expected_outgoing[counter], edge_index, edge);
            match expected_outgoing[counter] {
                (ref e, ref n) => {
                    assert_eq!(e, &edge.data);
                    assert_eq!(start_index, edge.source);
                    assert_eq!(n, graph.node_data(edge.target));
                }
            }
            counter += 1;
        }
        assert_eq!(counter, expected_outgoing.len());
    }

    #[test]
    fn each_adjacent_from_a() {
        let graph = create_graph();
        test_adjacent_edges(&graph, NodeIndex(0), "A",
                            [],
                            [("AB", "B")]);
    }

    #[test]
    fn each_adjacent_from_b() {
        let graph = create_graph();
        test_adjacent_edges(&graph, NodeIndex(1), "B",
                            [("FB", "F"), ("AB", "A"),],
                            [("BD", "D"), ("BC", "C"),]);
    }

    #[test]
    fn each_adjacent_from_c() {
        let graph = create_graph();
        test_adjacent_edges(&graph, NodeIndex(2), "C",
                            [("EC", "E"), ("BC", "B")],
                            []);
    }

    #[test]
    fn each_adjacent_from_d() {
        let graph = create_graph();
        test_adjacent_edges(&graph, NodeIndex(3), "D",
                            [("BD", "B")],
                            [("DE", "E")]);
    }
}