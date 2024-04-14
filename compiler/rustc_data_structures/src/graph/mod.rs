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

pub trait WithNumEdges: DirectedGraph {
    fn num_edges(&self) -> usize;
}

pub trait Successors: DirectedGraph {
    type Successors<'g>: Iterator<Item = Self::Node>
    where
        Self: 'g;

    fn successors(&self, node: Self::Node) -> Self::Successors<'_>;

    fn depth_first_search(&self, from: Self::Node) -> iterate::DepthFirstSearch<'_, Self> {
        iterate::DepthFirstSearch::new(self).with_start_node(from)
    }
}

pub trait Predecessors: DirectedGraph {
    type Predecessors<'g>: Iterator<Item = Self::Node>
    where
        Self: 'g;

    fn predecessors(&self, node: Self::Node) -> Self::Predecessors<'_>;
}

pub trait WithStartNode: DirectedGraph {
    fn start_node(&self) -> Self::Node;
}

pub trait ControlFlowGraph: DirectedGraph + WithStartNode + Predecessors + Successors {
    // convenient trait
}

impl<T> ControlFlowGraph for T where T: DirectedGraph + WithStartNode + Predecessors + Successors {}

/// Returns `true` if the graph has a cycle that is reachable from the start node.
pub fn is_cyclic<G>(graph: &G) -> bool
where
    G: ?Sized + DirectedGraph + WithStartNode + Successors,
{
    iterate::TriColorDepthFirstSearch::new(graph)
        .run_from_start(&mut iterate::CycleDetector)
        .is_some()
}
