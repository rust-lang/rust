use super::indexed_vec::Idx;

pub mod dominators;
pub mod implementation;
pub mod iterate;
mod reference;
pub mod scc;

#[cfg(test)]
mod test;

pub trait DirectedGraph {
    type Node: Idx;
}

pub trait WithNumNodes: DirectedGraph {
    fn num_nodes(&self) -> usize;
}

pub trait WithSuccessors: DirectedGraph
where
    Self: for<'graph> GraphSuccessors<'graph, Item = <Self as DirectedGraph>::Node>,
{
    fn successors<'graph>(
        &'graph self,
        node: Self::Node,
    ) -> <Self as GraphSuccessors<'graph>>::Iter;
}

pub trait GraphSuccessors<'graph> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
}

pub trait WithPredecessors: DirectedGraph
where
    Self: for<'graph> GraphPredecessors<'graph, Item = <Self as DirectedGraph>::Node>,
{
    fn predecessors<'graph>(
        &'graph self,
        node: Self::Node,
    ) -> <Self as GraphPredecessors<'graph>>::Iter;
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
