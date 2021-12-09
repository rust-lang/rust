//! A graph module for use in dataflow, region resolution, and elsewhere.

#![deny(missing_docs)]

use rustc_index::vec::Idx;

pub mod dominators;
pub mod implementation;
// TODO
pub mod iterate;
mod reference;
pub mod scc;
// TODO
pub mod vec_graph;

#[cfg(test)]
mod tests;

/// A directed graph.
pub trait DirectedGraph {
    #[allow(missing_docs)]
    type Node: Idx;
}

/// A directed graph with some number of nodes.
pub trait WithNumNodes: DirectedGraph {
    /// Returns the number of nodes in a graph.
    fn num_nodes(&self) -> usize;
}

/// A directed graph with some number of edges.
pub trait WithNumEdges: DirectedGraph {
    /// Returns the number of edges in a graph.
    fn num_edges(&self) -> usize;
}

/// A directed graph with successors.
pub trait WithSuccessors: DirectedGraph
where
    Self: for<'graph> GraphSuccessors<'graph, Item = <Self as DirectedGraph>::Node>,
{
    /// Returns an iterator of the successors for a node.
    fn successors(&self, node: Self::Node) -> <Self as GraphSuccessors<'_>>::Iter;

    #[allow(missing_docs)]
    fn depth_first_search(&self, from: Self::Node) -> iterate::DepthFirstSearch<'_, Self>
    where
        Self: WithNumNodes,
    {
        iterate::DepthFirstSearch::new(self).with_start_node(from)
    }
}

// TODO
#[allow(unused_lifetimes)]
pub trait GraphSuccessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}


// TODO
pub trait WithPredecessors: DirectedGraph
where
    Self: for<'graph> GraphPredecessors<'graph, Item = <Self as DirectedGraph>::Node>,
{
    fn predecessors(&self, node: Self::Node) -> <Self as GraphPredecessors<'_>>::Iter;
}

// TODO
#[allow(unused_lifetimes)]
pub trait GraphPredecessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}

// TODO
pub trait WithStartNode: DirectedGraph {
    fn start_node(&self) -> Self::Node;
}

// TODO
pub trait ControlFlowGraph:
    DirectedGraph + WithStartNode + WithPredecessors + WithSuccessors + WithNumNodes
{
    // convenient trait
}

impl<T> ControlFlowGraph for T where
    T: DirectedGraph + WithStartNode + WithPredecessors + WithSuccessors + WithNumNodes
{
}

/// Returns `true` if the graph has a cycle that is reachable from the start node.
pub fn is_cyclic<G>(graph: &G) -> bool
where
    G: ?Sized + DirectedGraph + WithStartNode + WithSuccessors + WithNumNodes,
{
    iterate::TriColorDepthFirstSearch::new(graph)
        .run_from_start(&mut iterate::CycleDetector)
        .is_some()
}
