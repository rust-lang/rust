use crate::graph::{DirectedGraph, Predecessors, Successors};

/// View that reverses the direction of edges in its underlying graph, so that
/// successors become predecessors and vice-versa.
///
/// Because of `impl<G: Graph> Graph for &G`, the underlying graph can be
/// wrapped by-reference instead of by-value if desired.
#[derive(Clone, Copy, Debug)]
pub struct ReversedGraph<G> {
    pub inner: G,
}

impl<G> ReversedGraph<G> {
    pub fn new(inner: G) -> Self {
        Self { inner }
    }
}

impl<G: DirectedGraph> DirectedGraph for ReversedGraph<G> {
    type Node = G::Node;

    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}

// Implementing `StartNode` is not possible in general, because the start node
// of an underlying graph is instead an _end_ node in the reversed graph.
// But would be possible to define another wrapper type that adds an explicit
// start node to its underlying graph, if desired.

impl<G: Predecessors> Successors for ReversedGraph<G> {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.inner.predecessors(node)
    }
}

impl<G: Successors> Predecessors for ReversedGraph<G> {
    fn predecessors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.inner.successors(node)
    }
}
