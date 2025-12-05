use rustc_index::Idx;

pub mod dominators;
pub mod iterate;
pub mod linked_graph;
mod reference;
pub mod reversed;
pub mod scc;
pub mod vec_graph;

#[cfg(test)]
mod tests;

pub trait DirectedGraph {
    type Node: Idx;

    /// Returns the total number of nodes in this graph.
    ///
    /// Several graph algorithm implementations assume that every node ID is
    /// strictly less than the number of nodes, i.e. nodes are densely numbered.
    /// That assumption allows them to use `num_nodes` to allocate per-node
    /// data structures, indexed by node.
    fn num_nodes(&self) -> usize;

    /// Iterates over all nodes of a graph in ascending numeric order.
    ///
    /// Assumes that nodes are densely numbered, i.e. every index in
    /// `0..num_nodes` is a valid node.
    fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = Self::Node> + DoubleEndedIterator + ExactSizeIterator {
        (0..self.num_nodes()).map(<Self::Node as Idx>::new)
    }
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

pub fn depth_first_search<G>(graph: G, from: G::Node) -> iterate::DepthFirstSearch<G>
where
    G: Successors,
{
    iterate::DepthFirstSearch::new(graph).with_start_node(from)
}

pub fn depth_first_search_as_undirected<G>(
    graph: G,
    from: G::Node,
) -> iterate::DepthFirstSearch<impl Successors<Node = G::Node>>
where
    G: Successors + Predecessors,
{
    struct AsUndirected<G>(G);

    impl<G: DirectedGraph> DirectedGraph for AsUndirected<G> {
        type Node = G::Node;

        fn num_nodes(&self) -> usize {
            self.0.num_nodes()
        }
    }

    impl<G: Successors + Predecessors> Successors for AsUndirected<G> {
        fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
            self.0.successors(node).chain(self.0.predecessors(node))
        }
    }

    iterate::DepthFirstSearch::new(AsUndirected(graph)).with_start_node(from)
}
