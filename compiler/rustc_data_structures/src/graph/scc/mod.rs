//! Routine to compute the strongly connected components (SCCs) of a graph.
//!
//! Also computes as the resulting DAG if each SCC is replaced with a
//! node in the graph. This uses [Tarjan's algorithm](
//! https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
//! that completes in *O*(*n*) time.

use crate::fx::FxHashSet;
use crate::graph::vec_graph::VecGraph;
use crate::graph::{DirectedGraph, GraphSuccessors, WithNumEdges, WithNumNodes, WithSuccessors};
use rustc_index::vec::{Idx, IndexVec};
use std::cmp::Ord;
use std::ops::Range;

#[cfg(test)]
mod tests;

/// Strongly connected components (SCC) of a graph. The type `N` is
/// the index type for the graph nodes and `S` is the index type for
/// the SCCs. We can map from each node to the SCC that it
/// participates in, and we also have the successors of each SCC.
pub struct Sccs<N: Idx, S: Idx> {
    /// For each node, what is the SCC index of the SCC to which it
    /// belongs.
    scc_indices: IndexVec<N, S>,

    /// Data about each SCC.
    scc_data: SccData<S>,
}

struct SccData<S: Idx> {
    /// For each SCC, the range of `all_successors` where its
    /// successors can be found.
    ranges: IndexVec<S, Range<usize>>,

    /// Contains the successors for all the Sccs, concatenated. The
    /// range of indices corresponding to a given SCC is found in its
    /// SccData.
    all_successors: Vec<S>,
}

impl<N: Idx, S: Idx + Ord> Sccs<N, S> {
    pub fn new(graph: &(impl DirectedGraph<Node = N> + WithNumNodes + WithSuccessors)) -> Self {
        SccsConstruction::construct(graph)
    }

    /// Returns the number of SCCs in the graph.
    pub fn num_sccs(&self) -> usize {
        self.scc_data.len()
    }

    /// Returns an iterator over the SCCs in the graph.
    ///
    /// The SCCs will be iterated in **dependency order** (or **post order**),
    /// meaning that if `S1 -> S2`, we will visit `S2` first and `S1` after.
    /// This is convenient when the edges represent dependencies: when you visit
    /// `S1`, the value for `S2` will already have been computed.
    pub fn all_sccs(&self) -> impl Iterator<Item = S> {
        (0..self.scc_data.len()).map(S::new)
    }

    /// Returns the SCC to which a node `r` belongs.
    pub fn scc(&self, r: N) -> S {
        self.scc_indices[r]
    }

    /// Returns the successors of the given SCC.
    pub fn successors(&self, scc: S) -> &[S] {
        self.scc_data.successors(scc)
    }

    /// Construct the reverse graph of the SCC graph.
    pub fn reverse(&self) -> VecGraph<S> {
        VecGraph::new(
            self.num_sccs(),
            self.all_sccs()
                .flat_map(|source| {
                    self.successors(source).iter().map(move |&target| (target, source))
                })
                .collect(),
        )
    }
}

impl<N: Idx, S: Idx> DirectedGraph for Sccs<N, S> {
    type Node = S;
}

impl<N: Idx, S: Idx + Ord> WithNumNodes for Sccs<N, S> {
    fn num_nodes(&self) -> usize {
        self.num_sccs()
    }
}

impl<N: Idx, S: Idx> WithNumEdges for Sccs<N, S> {
    fn num_edges(&self) -> usize {
        self.scc_data.all_successors.len()
    }
}

impl<'graph, N: Idx, S: Idx> GraphSuccessors<'graph> for Sccs<N, S> {
    type Item = S;

    type Iter = std::iter::Cloned<std::slice::Iter<'graph, S>>;
}

impl<N: Idx, S: Idx + Ord> WithSuccessors for Sccs<N, S> {
    fn successors(&self, node: S) -> <Self as GraphSuccessors<'_>>::Iter {
        self.successors(node).iter().cloned()
    }
}

impl<S: Idx> SccData<S> {
    /// Number of SCCs,
    fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Returns the successors of the given SCC.
    fn successors(&self, scc: S) -> &[S] {
        // Annoyingly, `range` does not implement `Copy`, so we have
        // to do `range.start..range.end`:
        let range = &self.ranges[scc];
        &self.all_successors[range.start..range.end]
    }

    /// Creates a new SCC with `successors` as its successors and
    /// returns the resulting index.
    fn create_scc(&mut self, successors: impl IntoIterator<Item = S>) -> S {
        // Store the successors on `scc_successors_vec`, remembering
        // the range of indices.
        let all_successors_start = self.all_successors.len();
        self.all_successors.extend(successors);
        let all_successors_end = self.all_successors.len();

        debug!(
            "create_scc({:?}) successors={:?}",
            self.ranges.len(),
            &self.all_successors[all_successors_start..all_successors_end],
        );

        self.ranges.push(all_successors_start..all_successors_end)
    }
}

struct SccsConstruction<'c, G: DirectedGraph + WithNumNodes + WithSuccessors, S: Idx> {
    graph: &'c G,

    /// The state of each node; used during walk to record the stack
    /// and after walk to record what cycle each node ended up being
    /// in.
    node_states: IndexVec<G::Node, NodeState<G::Node, S>>,

    /// The stack of nodes that we are visiting as part of the DFS.
    node_stack: Vec<G::Node>,

    /// The stack of successors: as we visit a node, we mark our
    /// position in this stack, and when we encounter a successor SCC,
    /// we push it on the stack. When we complete an SCC, we can pop
    /// everything off the stack that was found along the way.
    successors_stack: Vec<S>,

    /// A set used to strip duplicates. As we accumulate successors
    /// into the successors_stack, we sometimes get duplicate entries.
    /// We use this set to remove those -- we also keep its storage
    /// around between successors to amortize memory allocation costs.
    duplicate_set: FxHashSet<S>,

    scc_data: SccData<S>,
}

#[derive(Copy, Clone, Debug)]
enum NodeState<N, S> {
    /// This node has not yet been visited as part of the DFS.
    ///
    /// After SCC construction is complete, this state ought to be
    /// impossible.
    NotVisited,

    /// This node is currently being walk as part of our DFS. It is on
    /// the stack at the depth `depth`.
    ///
    /// After SCC construction is complete, this state ought to be
    /// impossible.
    BeingVisited { depth: usize },

    /// Indicates that this node is a member of the given cycle.
    InCycle { scc_index: S },

    /// Indicates that this node is a member of whatever cycle
    /// `parent` is a member of. This state is transient: whenever we
    /// see it, we try to overwrite it with the current state of
    /// `parent` (this is the "path compression" step of a union-find
    /// algorithm).
    InCycleWith { parent: N },
}

#[derive(Copy, Clone, Debug)]
enum WalkReturn<S> {
    Cycle { min_depth: usize },
    Complete { scc_index: S },
}

impl<'c, G, S> SccsConstruction<'c, G, S>
where
    G: DirectedGraph + WithNumNodes + WithSuccessors,
    S: Idx,
{
    /// Identifies SCCs in the graph `G` and computes the resulting
    /// DAG. This uses a variant of [Tarjan's
    /// algorithm][wikipedia]. The high-level summary of the algorithm
    /// is that we do a depth-first search. Along the way, we keep a
    /// stack of each node whose successors are being visited. We
    /// track the depth of each node on this stack (there is no depth
    /// if the node is not on the stack). When we find that some node
    /// N with depth D can reach some other node N' with lower depth
    /// D' (i.e., D' < D), we know that N, N', and all nodes in
    /// between them on the stack are part of an SCC.
    ///
    /// [wikipedia]: https://bit.ly/2EZIx84
    fn construct(graph: &'c G) -> Sccs<G::Node, S> {
        let num_nodes = graph.num_nodes();

        let mut this = Self {
            graph,
            node_states: IndexVec::from_elem_n(NodeState::NotVisited, num_nodes),
            node_stack: Vec::with_capacity(num_nodes),
            successors_stack: Vec::new(),
            scc_data: SccData { ranges: IndexVec::new(), all_successors: Vec::new() },
            duplicate_set: FxHashSet::default(),
        };

        let scc_indices = (0..num_nodes)
            .map(G::Node::new)
            .map(|node| match this.start_walk_from(node) {
                WalkReturn::Complete { scc_index } => scc_index,
                WalkReturn::Cycle { min_depth } => panic!(
                    "`start_walk_node({:?})` returned cycle with depth {:?}",
                    node, min_depth
                ),
            })
            .collect();

        Sccs { scc_indices, scc_data: this.scc_data }
    }

    fn start_walk_from(&mut self, node: G::Node) -> WalkReturn<S> {
        if let Some(result) = self.inspect_node(node) {
            result
        } else {
            self.walk_unvisited_node(node)
        }
    }

    /// Inspect a node during the DFS. We first examine its current
    /// state -- if it is not yet visited (`NotVisited`), return `None` so
    /// that the caller might push it onto the stack and start walking its
    /// successors.
    ///
    /// If it is already on the DFS stack it will be in the state
    /// `BeingVisited`. In that case, we have found a cycle and we
    /// return the depth from the stack.
    ///
    /// Otherwise, we are looking at a node that has already been
    /// completely visited. We therefore return `WalkReturn::Complete`
    /// with its associated SCC index.
    fn inspect_node(&mut self, node: G::Node) -> Option<WalkReturn<S>> {
        Some(match self.find_state(node) {
            NodeState::InCycle { scc_index } => WalkReturn::Complete { scc_index },

            NodeState::BeingVisited { depth: min_depth } => WalkReturn::Cycle { min_depth },

            NodeState::NotVisited => return None,

            NodeState::InCycleWith { parent } => panic!(
                "`find_state` returned `InCycleWith({:?})`, which ought to be impossible",
                parent
            ),
        })
    }

    /// Fetches the state of the node `r`. If `r` is recorded as being
    /// in a cycle with some other node `r2`, then fetches the state
    /// of `r2` (and updates `r` to reflect current result). This is
    /// basically the "find" part of a standard union-find algorithm
    /// (with path compression).
    fn find_state(&mut self, mut node: G::Node) -> NodeState<G::Node, S> {
        // To avoid recursion we temporarily reuse the `parent` of each
        // InCycleWith link to encode a downwards link while compressing
        // the path. After we have found the root or deepest node being
        // visited, we traverse the reverse links and correct the node
        // states on the way.
        //
        // **Note**: This mutation requires that this is a leaf function
        // or at least that none of the called functions inspects the
        // current node states. Luckily, we are a leaf.

        // Remember one previous link. The termination condition when
        // following links downwards is then simply as soon as we have
        // found the initial self-loop.
        let mut previous_node = node;

        // Ultimately assigned by the parent when following
        // `InCycleWith` upwards.
        let node_state = loop {
            debug!("find_state(r = {:?} in state {:?})", node, self.node_states[node]);
            match self.node_states[node] {
                NodeState::InCycle { scc_index } => break NodeState::InCycle { scc_index },
                NodeState::BeingVisited { depth } => break NodeState::BeingVisited { depth },
                NodeState::NotVisited => break NodeState::NotVisited,
                NodeState::InCycleWith { parent } => {
                    // We test this, to be extremely sure that we never
                    // ever break our termination condition for the
                    // reverse iteration loop.
                    assert!(node != parent, "Node can not be in cycle with itself");
                    // Store the previous node as an inverted list link
                    self.node_states[node] = NodeState::InCycleWith { parent: previous_node };
                    // Update to parent node.
                    previous_node = node;
                    node = parent;
                }
            }
        };

        // The states form a graph where up to one outgoing link is stored at
        // each node. Initially in general,
        //
        //                                                  E
        //                                                  ^
        //                                                  |
        //                                InCycleWith/BeingVisited/NotVisited
        //                                                  |
        //   A-InCycleWith->B-InCycleWith…>C-InCycleWith->D-+
        //   |
        //   = node, previous_node
        //
        // After the first loop, this will look like
        //                                                  E
        //                                                  ^
        //                                                  |
        //                                InCycleWith/BeingVisited/NotVisited
        //                                                  |
        // +>A<-InCycleWith-B<…InCycleWith-C<-InCycleWith-D-+
        // | |                             |              |
        // | InCycleWith                   |              = node
        // +-+                             =previous_node
        //
        // Note in particular that A will be linked to itself in a self-cycle
        // and no other self-cycles occur due to how InCycleWith is assigned in
        // the find phase implemented by `walk_unvisited_node`.
        //
        // We now want to compress the path, that is assign the state of the
        // link D-E to all other links.
        //
        // We can then walk backwards, starting from `previous_node`, and assign
        // each node in the list with the updated state. The loop terminates
        // when we reach the self-cycle.

        // Move backwards until we found the node where we started. We
        // will know when we hit the state where previous_node == node.
        loop {
            // Back at the beginning, we can return.
            if previous_node == node {
                return node_state;
            }
            // Update to previous node in the link.
            match self.node_states[previous_node] {
                NodeState::InCycleWith { parent: previous } => {
                    node = previous_node;
                    previous_node = previous;
                }
                // Only InCycleWith nodes were added to the reverse linked list.
                other => panic!("Invalid previous link while compressing cycle: {:?}", other),
            }

            debug!("find_state: parent_state = {:?}", node_state);

            // Update the node state from the parent state. The assigned
            // state is actually a loop invariant but it will only be
            // evaluated if there is at least one backlink to follow.
            // Fully trusting llvm here to find this loop optimization.
            match node_state {
                // Path compression, make current node point to the same root.
                NodeState::InCycle { .. } => {
                    self.node_states[node] = node_state;
                }
                // Still visiting nodes, compress to cycle to the node
                // at that depth.
                NodeState::BeingVisited { depth } => {
                    self.node_states[node] =
                        NodeState::InCycleWith { parent: self.node_stack[depth] };
                }
                // These are never allowed as parent nodes. InCycleWith
                // should have been followed to a real parent and
                // NotVisited can not be part of a cycle since it should
                // have instead gotten explored.
                NodeState::NotVisited | NodeState::InCycleWith { .. } => {
                    panic!("invalid parent state: {:?}", node_state)
                }
            }
        }
    }

    /// Walks a node that has never been visited before.
    ///
    /// Call this method when `inspect_node` has returned `None`. Having the
    /// caller decide avoids mutual recursion between the two methods and allows
    /// us to maintain an allocated stack for nodes on the path between calls.
    #[instrument(skip(self, initial), level = "debug")]
    fn walk_unvisited_node(&mut self, initial: G::Node) -> WalkReturn<S> {
        struct VisitingNodeFrame<G: DirectedGraph, Successors> {
            node: G::Node,
            iter: Option<Successors>,
            depth: usize,
            min_depth: usize,
            successors_len: usize,
            min_cycle_root: G::Node,
            successor_node: G::Node,
        }

        // Move the stack to a local variable. We want to utilize the existing allocation and
        // mutably borrow it without borrowing self at the same time.
        let mut successors_stack = core::mem::take(&mut self.successors_stack);
        debug_assert_eq!(successors_stack.len(), 0);

        let mut stack: Vec<VisitingNodeFrame<G, _>> = vec![VisitingNodeFrame {
            node: initial,
            depth: 0,
            min_depth: 0,
            iter: None,
            successors_len: 0,
            min_cycle_root: initial,
            successor_node: initial,
        }];

        let mut return_value = None;

        'recurse: while let Some(frame) = stack.last_mut() {
            let VisitingNodeFrame {
                node,
                depth,
                iter,
                successors_len,
                min_depth,
                min_cycle_root,
                successor_node,
            } = frame;

            let node = *node;
            let depth = *depth;

            let successors = match iter {
                Some(iter) => iter,
                None => {
                    // This None marks that we still have the initialize this node's frame.
                    debug!(?depth, ?node);

                    debug_assert!(matches!(self.node_states[node], NodeState::NotVisited));

                    // Push `node` onto the stack.
                    self.node_states[node] = NodeState::BeingVisited { depth };
                    self.node_stack.push(node);

                    // Walk each successor of the node, looking to see if any of
                    // them can reach a node that is presently on the stack. If
                    // so, that means they can also reach us.
                    *successors_len = successors_stack.len();
                    // Set and return a reference, this is currently empty.
                    iter.get_or_insert(self.graph.successors(node))
                }
            };

            // Now that iter is initialized, this is a constant for this frame.
            let successors_len = *successors_len;

            // Construct iterators for the nodes and walk results. There are two cases:
            // * The walk of a successor node returned.
            // * The remaining successor nodes.
            let returned_walk =
                return_value.take().into_iter().map(|walk| (*successor_node, Some(walk)));

            let successor_walk = successors.by_ref().map(|successor_node| {
                debug!(?node, ?successor_node);
                (successor_node, self.inspect_node(successor_node))
            });

            for (successor_node, walk) in returned_walk.chain(successor_walk) {
                match walk {
                    Some(WalkReturn::Cycle { min_depth: successor_min_depth }) => {
                        // Track the minimum depth we can reach.
                        assert!(successor_min_depth <= depth);
                        if successor_min_depth < *min_depth {
                            debug!(?node, ?successor_min_depth);
                            *min_depth = successor_min_depth;
                            *min_cycle_root = successor_node;
                        }
                    }

                    Some(WalkReturn::Complete { scc_index: successor_scc_index }) => {
                        // Push the completed SCC indices onto
                        // the `successors_stack` for later.
                        debug!(?node, ?successor_scc_index);
                        successors_stack.push(successor_scc_index);
                    }

                    None => {
                        let depth = depth + 1;
                        debug!(?depth, ?successor_node);
                        // Remember which node the return value will come from.
                        frame.successor_node = successor_node;
                        // Start a new stack frame the step into it.
                        stack.push(VisitingNodeFrame {
                            node: successor_node,
                            depth,
                            iter: None,
                            successors_len: 0,
                            min_depth: depth,
                            min_cycle_root: successor_node,
                            successor_node,
                        });
                        continue 'recurse;
                    }
                }
            }

            // Completed walk, remove `node` from the stack.
            let r = self.node_stack.pop();
            debug_assert_eq!(r, Some(node));

            // Remove the frame, it's done.
            let frame = stack.pop().unwrap();

            // If `min_depth == depth`, then we are the root of the
            // cycle: we can't reach anyone further down the stack.

            // Pass the 'return value' down the stack.
            // We return one frame at a time so there can't be another return value.
            debug_assert!(return_value.is_none());
            return_value = Some(if frame.min_depth == depth {
                // Note that successor stack may have duplicates, so we
                // want to remove those:
                let deduplicated_successors = {
                    let duplicate_set = &mut self.duplicate_set;
                    duplicate_set.clear();
                    successors_stack
                        .drain(successors_len..)
                        .filter(move |&i| duplicate_set.insert(i))
                };
                let scc_index = self.scc_data.create_scc(deduplicated_successors);
                self.node_states[node] = NodeState::InCycle { scc_index };
                WalkReturn::Complete { scc_index }
            } else {
                // We are not the head of the cycle. Return back to our
                // caller. They will take ownership of the
                // `self.successors` data that we pushed.
                self.node_states[node] = NodeState::InCycleWith { parent: frame.min_cycle_root };
                WalkReturn::Cycle { min_depth: frame.min_depth }
            });
        }

        // Keep the allocation we used for successors_stack.
        self.successors_stack = successors_stack;
        debug_assert_eq!(self.successors_stack.len(), 0);

        return_value.unwrap()
    }
}
