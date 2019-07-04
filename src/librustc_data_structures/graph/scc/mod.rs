//! Routine to compute the strongly connected components (SCCs) of a
//! graph, as well as the resulting DAG if each SCC is replaced with a
//! node in the graph. This uses Tarjan's algorithm that completes in
//! O(n) time.

use crate::fx::FxHashSet;
use crate::graph::{DirectedGraph, WithNumNodes, WithNumEdges, WithSuccessors, GraphSuccessors};
use crate::graph::vec_graph::VecGraph;
use crate::indexed_vec::{Idx, IndexVec};
use std::ops::Range;

mod test;

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

impl<N: Idx, S: Idx> Sccs<N, S> {
    pub fn new(graph: &(impl DirectedGraph<Node = N> + WithNumNodes + WithSuccessors)) -> Self {
        SccsConstruction::construct(graph)
    }

    /// Returns the number of SCCs in the graph.
    pub fn num_sccs(&self) -> usize {
        self.scc_data.len()
    }

    /// Returns an iterator over the SCCs in the graph.
    pub fn all_sccs(&self) -> impl Iterator<Item = S> {
        (0 .. self.scc_data.len()).map(S::new)
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
                .flat_map(|source| self.successors(source).iter().map(move |&target| {
                    (target, source)
                }))
                .collect(),
        )
    }
}

impl<N: Idx, S: Idx> DirectedGraph for Sccs<N, S> {
    type Node = S;
}

impl<N: Idx, S: Idx> WithNumNodes for Sccs<N, S> {
    fn num_nodes(&self) -> usize {
        self.num_sccs()
    }
}

impl<N: Idx, S: Idx> WithNumEdges for Sccs<N, S> {
    fn num_edges(&self) -> usize {
        self.scc_data.all_successors.len()
    }
}

impl<N: Idx, S: Idx> GraphSuccessors<'graph> for Sccs<N, S> {
    type Item = S;

    type Iter = std::iter::Cloned<std::slice::Iter<'graph, S>>;
}

impl<N: Idx, S: Idx> WithSuccessors for Sccs<N, S> {
    fn successors<'graph>(
        &'graph self,
        node: S
    ) -> <Self as GraphSuccessors<'graph>>::Iter {
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
            scc_data: SccData {
                ranges: IndexVec::new(),
                all_successors: Vec::new(),
            },
            duplicate_set: FxHashSet::default(),
        };

        let scc_indices = (0..num_nodes)
            .map(G::Node::new)
            .map(|node| match this.walk_node(0, node) {
                WalkReturn::Complete { scc_index } => scc_index,
                WalkReturn::Cycle { min_depth } => panic!(
                    "`walk_node(0, {:?})` returned cycle with depth {:?}",
                    node, min_depth
                ),
            })
            .collect();

        Sccs {
            scc_indices,
            scc_data: this.scc_data,
        }
    }

    /// Visits a node during the DFS. We first examine its current
    /// state -- if it is not yet visited (`NotVisited`), we can push
    /// it onto the stack and start walking its successors.
    ///
    /// If it is already on the DFS stack it will be in the state
    /// `BeingVisited`. In that case, we have found a cycle and we
    /// return the depth from the stack.
    ///
    /// Otherwise, we are looking at a node that has already been
    /// completely visited. We therefore return `WalkReturn::Complete`
    /// with its associated SCC index.
    fn walk_node(&mut self, depth: usize, node: G::Node) -> WalkReturn<S> {
        debug!("walk_node(depth = {:?}, node = {:?})", depth, node);
        match self.find_state(node) {
            NodeState::InCycle { scc_index } => WalkReturn::Complete { scc_index },

            NodeState::BeingVisited { depth: min_depth } => WalkReturn::Cycle { min_depth },

            NodeState::NotVisited => self.walk_unvisited_node(depth, node),

            NodeState::InCycleWith { parent } => panic!(
                "`find_state` returned `InCycleWith({:?})`, which ought to be impossible",
                parent
            ),
        }
    }

    /// Fetches the state of the node `r`. If `r` is recorded as being
    /// in a cycle with some other node `r2`, then fetches the state
    /// of `r2` (and updates `r` to reflect current result). This is
    /// basically the "find" part of a standard union-find algorithm
    /// (with path compression).
    fn find_state(&mut self, r: G::Node) -> NodeState<G::Node, S> {
        debug!("find_state(r = {:?} in state {:?})", r, self.node_states[r]);
        match self.node_states[r] {
            NodeState::InCycle { scc_index } => NodeState::InCycle { scc_index },
            NodeState::BeingVisited { depth } => NodeState::BeingVisited { depth },
            NodeState::NotVisited => NodeState::NotVisited,
            NodeState::InCycleWith { parent } => {
                let parent_state = self.find_state(parent);
                debug!("find_state: parent_state = {:?}", parent_state);
                match parent_state {
                    NodeState::InCycle { .. } => {
                        self.node_states[r] = parent_state;
                        parent_state
                    }

                    NodeState::BeingVisited { depth } => {
                        self.node_states[r] = NodeState::InCycleWith {
                            parent: self.node_stack[depth],
                        };
                        parent_state
                    }

                    NodeState::NotVisited | NodeState::InCycleWith { .. } => {
                        panic!("invalid parent state: {:?}", parent_state)
                    }
                }
            }
        }
    }

    /// Walks a node that has never been visited before.
    fn walk_unvisited_node(&mut self, depth: usize, node: G::Node) -> WalkReturn<S> {
        debug!(
            "walk_unvisited_node(depth = {:?}, node = {:?})",
            depth, node
        );

        debug_assert!(match self.node_states[node] {
            NodeState::NotVisited => true,
            _ => false,
        });

        // Push `node` onto the stack.
        self.node_states[node] = NodeState::BeingVisited { depth };
        self.node_stack.push(node);

        // Walk each successor of the node, looking to see if any of
        // them can reach a node that is presently on the stack. If
        // so, that means they can also reach us.
        let mut min_depth = depth;
        let mut min_cycle_root = node;
        let successors_len = self.successors_stack.len();
        for successor_node in self.graph.successors(node) {
            debug!(
                "walk_unvisited_node: node = {:?} successor_ode = {:?}",
                node, successor_node
            );
            match self.walk_node(depth + 1, successor_node) {
                WalkReturn::Cycle {
                    min_depth: successor_min_depth,
                } => {
                    // Track the minimum depth we can reach.
                    assert!(successor_min_depth <= depth);
                    if successor_min_depth < min_depth {
                        debug!(
                            "walk_unvisited_node: node = {:?} successor_min_depth = {:?}",
                            node, successor_min_depth
                        );
                        min_depth = successor_min_depth;
                        min_cycle_root = successor_node;
                    }
                }

                WalkReturn::Complete {
                    scc_index: successor_scc_index,
                } => {
                    // Push the completed SCC indices onto
                    // the `successors_stack` for later.
                    debug!(
                        "walk_unvisited_node: node = {:?} successor_scc_index = {:?}",
                        node, successor_scc_index
                    );
                    self.successors_stack.push(successor_scc_index);
                }
            }
        }

        // Completed walk, remove `node` from the stack.
        let r = self.node_stack.pop();
        debug_assert_eq!(r, Some(node));

        // If `min_depth == depth`, then we are the root of the
        // cycle: we can't reach anyone further down the stack.
        if min_depth == depth {
            // Note that successor stack may have duplicates, so we
            // want to remove those:
            let deduplicated_successors = {
                let duplicate_set = &mut self.duplicate_set;
                duplicate_set.clear();
                self.successors_stack
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
            self.node_states[node] = NodeState::InCycleWith {
                parent: min_cycle_root,
            };
            WalkReturn::Cycle { min_depth }
        }
    }
}
