use rustc_data_structures::graph;
use rustc_index::Idx;

pub(crate) trait IterNodes: graph::DirectedGraph {
    /// Iterates over all nodes of a graph in ascending numeric order.
    /// Assumes that nodes are densely numbered, i.e. every index in
    /// `0..num_nodes` is a valid node.
    ///
    /// FIXME: Can this just be part of [`graph::DirectedGraph`]?
    fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = Self::Node> + DoubleEndedIterator + ExactSizeIterator {
        (0..self.num_nodes()).map(<Self::Node as Idx>::new)
    }
}
impl<G: graph::DirectedGraph> IterNodes for G {}
