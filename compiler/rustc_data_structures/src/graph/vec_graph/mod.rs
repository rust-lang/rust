use std::cmp::Ord;

use crate::graph::{DirectedGraph, GraphSuccessors, WithNumEdges, WithNumNodes, WithSuccessors};
use rustc_index::vec::{Idx, IndexVec};

#[cfg(test)]
mod tests;

pub struct VecGraph<N: Idx> {
    /// Maps from a given node to an index where the set of successors
    /// for that node starts. The index indexes into the `edges`
    /// vector. To find the range for a given node, we look up the
    /// start for that node and then the start for the next node
    /// (i.e., with an index 1 higher) and get the range between the
    /// two. This vector always has an extra entry so that this works
    /// even for the max element.
    node_starts: IndexVec<N, usize>,

    edge_targets: Vec<N>,
}

impl<N: Idx + Ord> VecGraph<N> {
    pub fn new(num_nodes: usize, mut edge_pairs: Vec<(N, N)>) -> Self {
        // Sort the edges by the source -- this is important.
        edge_pairs.sort();

        let num_edges = edge_pairs.len();

        // Store the *target* of each edge into `edge_targets`.
        let edge_targets: Vec<N> = edge_pairs.iter().map(|&(_, target)| target).collect();

        // Create the *edge starts* array. We are iterating over over
        // the (sorted) edge pairs. We maintain the invariant that the
        // length of the `node_starts` array is enough to store the
        // current source node -- so when we see that the source node
        // for an edge is greater than the current length, we grow the
        // edge-starts array by just enough.
        let mut node_starts = IndexVec::with_capacity(num_edges);
        for (index, &(source, _)) in edge_pairs.iter().enumerate() {
            // If we have a list like `[(0, x), (2, y)]`:
            //
            // - Start out with `node_starts` of `[]`
            // - Iterate to `(0, x)` at index 0:
            //   - Push one entry because `node_starts.len()` (0) is <= the source (0)
            //   - Leaving us with `node_starts` of `[0]`
            // - Iterate to `(2, y)` at index 1:
            //   - Push one entry because `node_starts.len()` (1) is <= the source (2)
            //   - Push one entry because `node_starts.len()` (2) is <= the source (2)
            //   - Leaving us with `node_starts` of `[0, 1, 1]`
            // - Loop terminates
            while node_starts.len() <= source.index() {
                node_starts.push(index);
            }
        }

        // Pad out the `node_starts` array so that it has `num_nodes +
        // 1` entries. Continuing our example above, if `num_nodes` is
        // be `3`, we would push one more index: `[0, 1, 1, 2]`.
        //
        // Interpretation of that vector:
        //
        // [0, 1, 1, 2]
        //        ---- range for N=2
        //     ---- range for N=1
        //  ---- range for N=0
        while node_starts.len() <= num_nodes {
            node_starts.push(edge_targets.len());
        }

        assert_eq!(node_starts.len(), num_nodes + 1);

        Self { node_starts, edge_targets }
    }

    /// Gets the successors for `source` as a slice.
    pub fn successors(&self, source: N) -> &[N] {
        let start_index = self.node_starts[source];
        let end_index = self.node_starts[source.plus(1)];
        &self.edge_targets[start_index..end_index]
    }
}

impl<N: Idx> DirectedGraph for VecGraph<N> {
    type Node = N;
}

impl<N: Idx> WithNumNodes for VecGraph<N> {
    fn num_nodes(&self) -> usize {
        self.node_starts.len() - 1
    }
}

impl<N: Idx> WithNumEdges for VecGraph<N> {
    fn num_edges(&self) -> usize {
        self.edge_targets.len()
    }
}

impl<'graph, N: Idx> GraphSuccessors<'graph> for VecGraph<N> {
    type Item = N;

    type Iter = std::iter::Cloned<std::slice::Iter<'graph, N>>;
}

impl<N: Idx + Ord> WithSuccessors for VecGraph<N> {
    fn successors(&self, node: N) -> <Self as GraphSuccessors<'_>>::Iter {
        self.successors(node).iter().cloned()
    }
}
