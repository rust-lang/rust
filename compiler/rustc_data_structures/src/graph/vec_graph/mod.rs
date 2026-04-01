use rustc_index::{Idx, IndexVec};

use crate::graph::{DirectedGraph, NumEdges, Predecessors, Successors};

#[cfg(test)]
mod tests;

/// A directed graph, efficient for cases where node indices are pre-existing.
///
/// If `BR` is true, the graph will store back-references, allowing you to get predecessors.
pub struct VecGraph<N: Idx, const BR: bool = false> {
    // This is basically a `HashMap<N, (Vec<N>, If<BR, Vec<N>>)>` -- a map from a node index, to
    // a list of targets of outgoing edges and (if enabled) a list of sources of incoming edges.
    //
    // However, it is condensed into two arrays as an optimization.
    //
    // `node_starts[n]` is the start of the list of targets of outgoing edges for node `n`.
    // So you can get node's successors with `edge_targets[node_starts[n]..node_starts[n + 1]]`.
    //
    // If `BR` is true (back references are enabled), then `node_starts[n + edge_count]` is the
    // start of the list of *sources* of incoming edges. You can get predecessors of a node
    // similarly to its successors but offsetting by `edge_count`. `edge_count` is
    // `edge_targets.len()/2` (again, in case BR is true) because half of the vec is back refs.
    //
    // All of this might be confusing, so here is an example graph and its representation:
    //
    //       n3 ----+
    //        ^     |                           (if BR = true)
    //        |     v     outgoing edges        incoming edges
    // n0 -> n1 -> n2     ______________      __________________
    //                   /              \    /                  \
    //  node indices[1]:  n0, n1, n2, n3,     n0, n1, n2, n3,       n/a
    //      vec indices:  n0, n1, n2, n3,     n4, n5, n6, n7,       n8
    //      node_starts:  [0,  1,  3,  4       4,  4,  5,  7,        8]
    //                     |   |   |   |       |   |   |   |         |
    //                     |   |   +---+       +---+   |   +---+     |
    //                     |   |       |           |   |       |     |
    //                     v   v       v           v   v       v     v
    //     edge_targets: [n1, n2, n3, n2          n0, n1, n3, n1]
    //                   /    \____/   |           |  \____/    \
    //             n0->n1     /        |           |       \     n3<-n1
    //                       /    n3->n2 [2]  n1<-n0 [2]    \
    //         n1->n2, n1->n3                                n2<-n1, n2<-n3
    //
    // The incoming edges are basically stored in the same way as outgoing edges, but offset and
    // the graph they store is the inverse of the original. Last index in the `node_starts` array
    // always points to one-past-the-end, so that we don't need to bound check `node_starts[n + 1]`
    //
    // [1]: "node indices" are the indices a user of `VecGraph` might use,
    //      note that they are different from "vec indices",
    //      which are the real indices you need to index `node_starts`
    //
    // [2]: Note that even though n2 also points to here,
    //      the next index also points here, so n2 has no
    //      successors (`edge_targets[3..3] = []`).
    //      Similarly with n0 and incoming edges
    //
    // If this is still confusing... then sorry :(
    //
    /// Indices into `edge_targets` that signify a start of list of edges.
    node_starts: IndexVec<N, usize>,

    /// Targets (or sources for back refs) of edges
    edge_targets: Vec<N>,
}

impl<N: Idx + Ord, const BR: bool> VecGraph<N, BR> {
    pub fn new(num_nodes: usize, mut edge_pairs: Vec<(N, N)>) -> Self {
        let num_edges = edge_pairs.len();

        let nodes_cap = match BR {
            // +1 for special entry at the end, pointing one past the end of `edge_targets`
            false => num_nodes + 1,
            // *2 for back references
            true => (num_nodes * 2) + 1,
        };

        let edges_cap = match BR {
            false => num_edges,
            // *2 for back references
            true => num_edges * 2,
        };

        let mut node_starts = IndexVec::with_capacity(nodes_cap);
        let mut edge_targets = Vec::with_capacity(edges_cap);

        // Sort the edges by the source -- this is important.
        edge_pairs.sort();

        // Fill forward references
        create_index(
            num_nodes,
            &mut edge_pairs.iter().map(|&(src, _)| src),
            &mut edge_pairs.iter().map(|&(_, tgt)| tgt),
            &mut edge_targets,
            &mut node_starts,
        );

        // Fill back references
        if BR {
            // Pop the special "last" entry, it will be replaced by first back ref
            node_starts.pop();

            // Re-sort the edges so that they are sorted by target
            edge_pairs.sort_by_key(|&(src, tgt)| (tgt, src));

            create_index(
                // Back essentially double the number of nodes
                num_nodes * 2,
                // NB: the source/target are switched here too
                // NB: we double the key index, so that we can later use *2 to get the back references
                &mut edge_pairs.iter().map(|&(_, tgt)| N::new(tgt.index() + num_nodes)),
                &mut edge_pairs.iter().map(|&(src, _)| src),
                &mut edge_targets,
                &mut node_starts,
            );
        }

        Self { node_starts, edge_targets }
    }

    /// Gets the successors for `source` as a slice.
    pub fn successors(&self, source: N) -> &[N] {
        assert!(source.index() < self.num_nodes());

        let start_index = self.node_starts[source];
        let end_index = self.node_starts[source.plus(1)];
        &self.edge_targets[start_index..end_index]
    }
}

impl<N: Idx + Ord> VecGraph<N, true> {
    /// Gets the predecessors for `target` as a slice.
    pub fn predecessors(&self, target: N) -> &[N] {
        assert!(target.index() < self.num_nodes());

        let target = N::new(target.index() + self.num_nodes());

        let start_index = self.node_starts[target];
        let end_index = self.node_starts[target.plus(1)];
        &self.edge_targets[start_index..end_index]
    }
}

/// Creates/initializes the index for the [`VecGraph`]. A helper for [`VecGraph::new`].
///
/// - `num_nodes` is the target number of nodes in the graph
/// - `sorted_edge_sources` are the edge sources, sorted
/// - `associated_edge_targets` are the edge *targets* in the same order as sources
/// - `edge_targets` is the vec of targets to be extended
/// - `node_starts` is the index to be filled
fn create_index<N: Idx + Ord>(
    num_nodes: usize,
    sorted_edge_sources: &mut dyn Iterator<Item = N>,
    associated_edge_targets: &mut dyn Iterator<Item = N>,
    edge_targets: &mut Vec<N>,
    node_starts: &mut IndexVec<N, usize>,
) {
    let offset = edge_targets.len();

    // Store the *target* of each edge into `edge_targets`.
    edge_targets.extend(associated_edge_targets);

    // Create the *edge starts* array. We are iterating over the
    // (sorted) edge pairs. We maintain the invariant that the
    // length of the `node_starts` array is enough to store the
    // current source node -- so when we see that the source node
    // for an edge is greater than the current length, we grow the
    // edge-starts array by just enough.
    for (index, source) in sorted_edge_sources.enumerate() {
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
            node_starts.push(index + offset);
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
}

impl<N: Idx, const BR: bool> DirectedGraph for VecGraph<N, BR> {
    type Node = N;

    fn num_nodes(&self) -> usize {
        match BR {
            false => self.node_starts.len() - 1,
            // If back refs are enabled, half of the array is said back refs
            true => (self.node_starts.len() - 1) / 2,
        }
    }
}

impl<N: Idx, const BR: bool> NumEdges for VecGraph<N, BR> {
    fn num_edges(&self) -> usize {
        match BR {
            false => self.edge_targets.len(),
            // If back refs are enabled, half of the array is reversed edges for them
            true => self.edge_targets.len() / 2,
        }
    }
}

impl<N: Idx + Ord, const BR: bool> Successors for VecGraph<N, BR> {
    fn successors(&self, node: N) -> impl Iterator<Item = Self::Node> {
        self.successors(node).iter().cloned()
    }
}

impl<N: Idx + Ord> Predecessors for VecGraph<N, true> {
    fn predecessors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.predecessors(node).iter().cloned()
    }
}
