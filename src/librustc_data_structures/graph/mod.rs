use super::indexed_vec::Idx;

pub mod dominators;
pub mod implementation;
pub mod iterate;
mod reference;
pub mod scc;
pub mod vec_graph;

#[cfg(test)]
mod test;

pub trait DirectedGraph {
    type Node: Idx;
}

pub trait WithNumNodes: DirectedGraph {
    fn num_nodes(&self) -> usize;
}

pub trait WithNumEdges: DirectedGraph {
    fn num_edges(&self) -> usize;
}

pub trait WithSuccessors: DirectedGraph
where
    Self: for<'graph> GraphSuccessors<'graph, Item = <Self as DirectedGraph>::Node>,
{
    fn successors(
        &self,
        node: Self::Node,
    ) -> <Self as GraphSuccessors<'_>>::Iter;

    fn depth_first_search(&self, from: Self::Node) -> iterate::DepthFirstSearch<'_, Self>
    where
        Self: WithNumNodes,
    {
        iterate::DepthFirstSearch::new(self, from)
    }
}

pub trait GraphSuccessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}

pub trait WithPredecessors: DirectedGraph
where
    Self: for<'graph> GraphPredecessors<'graph, Item = <Self as DirectedGraph>::Node>,
{
    fn predecessors(
        &self,
        node: Self::Node,
    ) -> <Self as GraphPredecessors<'_>>::Iter;
}

pub trait GraphPredecessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}

pub trait WithStartNode: DirectedGraph {
    fn start_node(&self) -> Self::Node;
}

pub trait ControlFlowGraph:
    DirectedGraph + WithStartNode + WithPredecessors + WithStartNode + WithSuccessors + WithNumNodes
{
    // convenient trait
}

impl<T> ControlFlowGraph for T
where
    T: DirectedGraph
        + WithStartNode
        + WithPredecessors
        + WithStartNode
        + WithSuccessors
        + WithNumNodes,
{
}
