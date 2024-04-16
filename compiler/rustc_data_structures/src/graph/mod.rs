use rustc_index::Idx;

pub mod dominators;
pub mod implementation;
pub mod iterate;
mod reference;
pub mod scc;
pub mod vec_graph;

#[cfg(test)]
mod tests;

pub trait DirectedGraph {
    type Node: Idx;

    fn num_nodes(&self) -> usize;
}

pub trait NumEdges: DirectedGraph {
    fn num_edges(&self) -> usize;
}

pub trait StartNode: DirectedGraph {
    fn start_node(&self) -> Self::Node;
}

pub trait Successors: DirectedGraph {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node>;
}

pub trait Predecessors: DirectedGraph {
    fn predecessors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node>;
}

/// Alias for [`DirectedGraph`] + [`StartNode`] + [`Predecessors`] + [`Successors`].
pub trait ControlFlowGraph: DirectedGraph + StartNode + Predecessors + Successors {}
impl<T> ControlFlowGraph for T where T: DirectedGraph + StartNode + Predecessors + Successors {}

/// Returns `true` if the graph has a cycle that is reachable from the start node.
pub fn is_cyclic<G>(graph: &G) -> bool
where
    G: ?Sized + DirectedGraph + StartNode + Successors,
{
    iterate::TriColorDepthFirstSearch::new(graph)
        .run_from_start(&mut iterate::CycleDetector)
        .is_some()
}

pub fn depth_first_search<G>(graph: &G, from: G::Node) -> iterate::DepthFirstSearch<'_, G>
where
    G: ?Sized + Successors,
{
    iterate::DepthFirstSearch::new(graph).with_start_node(from)
}
