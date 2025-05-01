//! Routine to compute the strongly connected components (SCCs) of a graph.
//!
//! Also computes as the resulting DAG if each SCC is replaced with a
//! node in the graph. This uses [Tarjan's algorithm](
//! https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
//! that completes in *O*(*n*) time.
//! Optionally, also annotate the SCC nodes with some commutative data.
//! Typical examples would include: minimum element in SCC, maximum element
//! reachable from it, etc.

use std::assert_matches::debug_assert_matches;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;

use rustc_index::{Idx, IndexSlice, IndexVec};
use tracing::{debug, instrument, trace};

use crate::fx::FxHashSet;
use crate::graph::vec_graph::VecGraph;
use crate::graph::{DirectedGraph, NumEdges, Successors};

#[cfg(test)]
mod tests;

/// An annotation for an SCC. This can be a representative,
/// the max/min element of the SCC, or all of the above.
///
/// Concretely, the both merge operations must commute, e.g. where `merge`
/// is `merge_scc` and `merge_reached`: `a.merge(b) == b.merge(a)`
///
/// In general, what you want is probably always min/max according
/// to some ordering, potentially with side constraints (min x such
/// that P holds).
pub trait Annotation: Debug + Copy {
    /// Merge two existing annotations into one during
    /// path compression.o
    fn merge_scc(self, other: Self) -> Self;

    /// Merge a successor into this annotation.
    fn merge_reached(self, other: Self) -> Self;

    fn update_scc(&mut self, other: Self) {
        *self = self.merge_scc(other)
    }

    fn update_reachable(&mut self, other: Self) {
        *self = self.merge_reached(other)
    }
}

/// An accumulator for annotations.
pub trait Annotations<N: Idx> {
    type Ann: Annotation;
    type SccIdx: Idx + Ord;

    fn new(&self, element: N) -> Self::Ann;
    fn annotate_scc(&mut self, scc: Self::SccIdx, annotation: Self::Ann);
}

/// The nil annotation accumulator, which does nothing.
struct NoAnnotations<S: Idx + Ord>(PhantomData<S>);

impl<N: Idx, S: Idx + Ord> Annotations<N> for NoAnnotations<S> {
    type SccIdx = S;
    type Ann = ();
    fn new(&self, _element: N) {}
    fn annotate_scc(&mut self, _scc: S, _annotation: ()) {}
}

/// The empty annotation, which does nothing.
impl Annotation for () {
    fn merge_reached(self, _other: Self) -> Self {
        ()
    }
    fn merge_scc(self, _other: Self) -> Self {
        ()
    }
}

/// Strongly connected components (SCC) of a graph. The type `N` is
/// the index type for the graph nodes and `S` is the index type for
/// the SCCs. We can map from each node to the SCC that it
/// participates in, and we also have the successors of each SCC.
pub struct Sccs<N: Idx, S: Idx> {
    /// For each node, what is the SCC index of the SCC to which it
    /// belongs.
    scc_indices: IndexVec<N, S>,

    /// Data about all the SCCs.
    scc_data: SccData<S>,
}

/// Information about an invidividual SCC node.
struct SccDetails {
    /// For this SCC, the range of `all_successors` where its
    /// successors can be found.
    range: Range<usize>,
}

// The name of this struct should discourage you from making it public and leaking
// its representation. This message was left here by one who came before you,
// who learnt the hard way that making even small changes in representation
// is difficult when it's publicly inspectable.
//
// Obey the law of Demeter!
struct SccData<S: Idx> {
    /// Maps SCC indices to their metadata, including
    /// offsets into `all_successors`.
    scc_details: IndexVec<S, SccDetails>,

    /// Contains the successors for all the Sccs, concatenated. The
    /// range of indices corresponding to a given SCC is found in its
    /// `scc_details.range`.
    all_successors: Vec<S>,
}

impl<N: Idx, S: Idx + Ord> Sccs<N, S> {
    /// Compute SCCs without annotations.
    pub fn new(graph: &impl Successors<Node = N>) -> Self {
        Self::new_with_annotation(graph, &mut NoAnnotations(PhantomData::<S>))
    }

    /// Compute SCCs and annotate them with a user-supplied annotation
    pub fn new_with_annotation<A: Annotations<N, SccIdx = S>>(
        graph: &impl Successors<Node = N>,
        annotations: &mut A,
    ) -> Self {
        SccsConstruction::construct(graph, annotations)
    }

    pub fn scc_indices(&self) -> &IndexSlice<N, S> {
        &self.scc_indices
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
    pub fn all_sccs(&self) -> impl Iterator<Item = S> + 'static {
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

impl<N: Idx, S: Idx + Ord> DirectedGraph for Sccs<N, S> {
    type Node = S;

    fn num_nodes(&self) -> usize {
        self.num_sccs()
    }
}

impl<N: Idx, S: Idx + Ord> NumEdges for Sccs<N, S> {
    fn num_edges(&self) -> usize {
        self.scc_data.all_successors.len()
    }
}

impl<N: Idx, S: Idx + Ord> Successors for Sccs<N, S> {
    fn successors(&self, node: S) -> impl Iterator<Item = Self::Node> {
        self.successors(node).iter().cloned()
    }
}

impl<S: Idx> SccData<S> {
    /// Number of SCCs,
    fn len(&self) -> usize {
        self.scc_details.len()
    }

    /// Returns the successors of the given SCC.
    fn successors(&self, scc: S) -> &[S] {
        &self.all_successors[self.scc_details[scc].range.clone()]
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
            self.len(),
            &self.all_successors[all_successors_start..all_successors_end],
        );

        let range = all_successors_start..all_successors_end;
        let metadata = SccDetails { range };
        self.scc_details.push(metadata)
    }
}

struct SccsConstruction<'c, 'a, G, A>
where
    G: DirectedGraph + Successors,
    A: Annotations<G::Node>,
{
    graph: &'c G,

    /// The state of each node; used during walk to record the stack
    /// and after walk to record what cycle each node ended up being
    /// in.
    node_states: IndexVec<G::Node, NodeState<G::Node, A::SccIdx, A::Ann>>,

    /// The stack of nodes that we are visiting as part of the DFS.
    node_stack: Vec<G::Node>,

    /// The stack of successors: as we visit a node, we mark our
    /// position in this stack, and when we encounter a successor SCC,
    /// we push it on the stack. When we complete an SCC, we can pop
    /// everything off the stack that was found along the way.
    successors_stack: Vec<A::SccIdx>,

    /// A set used to strip duplicates. As we accumulate successors
    /// into the successors_stack, we sometimes get duplicate entries.
    /// We use this set to remove those -- we also keep its storage
    /// around between successors to amortize memory allocation costs.
    duplicate_set: FxHashSet<A::SccIdx>,

    scc_data: SccData<A::SccIdx>,

    annotations: &'a mut A,
}

#[derive(Copy, Clone, Debug)]
enum NodeState<N, S, A: Annotation> {
    /// This node has not yet been visited as part of the DFS.
    ///
    /// After SCC construction is complete, this state ought to be
    /// impossible.
    NotVisited,

    /// This node is currently being walked as part of our DFS. It is on
    /// the stack at the depth `depth` and its current annotation is
    /// `annotation`.
    ///
    /// After SCC construction is complete, this state ought to be
    /// impossible.
    BeingVisited { depth: usize, annotation: A },

    /// Indicates that this node is a member of the given cycle where
    /// the merged annotation is `annotation`.
    /// Note that an SCC can have several cycles, so its final annotation
    /// is the merged value of all its member annotations.
    InCycle { scc_index: S, annotation: A },

    /// Indicates that this node is a member of whatever cycle
    /// `parent` is a member of. This state is transient: whenever we
    /// see it, we try to overwrite it with the current state of
    /// `parent` (this is the "path compression" step of a union-find
    /// algorithm).
    InCycleWith { parent: N },
}

/// The state of walking a given node.
#[derive(Copy, Clone, Debug)]
enum WalkReturn<S, A: Annotation> {
    /// The walk found a cycle, but the entire component is not known to have
    /// been fully walked yet. We only know the minimum depth of  this
    /// component in a minimum spanning tree of the graph. This component
    /// is tentatively represented by the state of the first node of this
    /// cycle we met, which is at `min_depth`.
    Cycle { min_depth: usize, annotation: A },
    /// The SCC and everything reachable from it have been fully walked.
    /// At this point we know what is inside the SCC as we have visited every
    /// node reachable from it. The SCC can now be fully represented by its ID.
    Complete { scc_index: S, annotation: A },
}

impl<'c, 'a, G, A> SccsConstruction<'c, 'a, G, A>
where
    G: DirectedGraph + Successors,
    A: Annotations<G::Node>,
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
    /// Additionally, we keep track of a current annotation of the SCC.
    ///
    /// [wikipedia]: https://bit.ly/2EZIx84
    fn construct(graph: &'c G, annotations: &'a mut A) -> Sccs<G::Node, A::SccIdx> {
        let num_nodes = graph.num_nodes();

        let mut this = Self {
            graph,
            node_states: IndexVec::from_elem_n(NodeState::NotVisited, num_nodes),
            node_stack: Vec::with_capacity(num_nodes),
            successors_stack: Vec::new(),
            scc_data: SccData { scc_details: IndexVec::new(), all_successors: Vec::new() },
            duplicate_set: FxHashSet::default(),
            annotations,
        };

        let scc_indices = graph
            .iter_nodes()
            .map(|node| match this.start_walk_from(node) {
                WalkReturn::Complete { scc_index, .. } => scc_index,
                WalkReturn::Cycle { min_depth, .. } => {
                    panic!("`start_walk_node({node:?})` returned cycle with depth {min_depth:?}")
                }
            })
            .collect();

        Sccs { scc_indices, scc_data: this.scc_data }
    }

    fn start_walk_from(&mut self, node: G::Node) -> WalkReturn<A::SccIdx, A::Ann> {
        self.inspect_node(node).unwrap_or_else(|| self.walk_unvisited_node(node))
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
    fn inspect_node(&mut self, node: G::Node) -> Option<WalkReturn<A::SccIdx, A::Ann>> {
        Some(match self.find_state(node) {
            NodeState::InCycle { scc_index, annotation } => {
                WalkReturn::Complete { scc_index, annotation }
            }

            NodeState::BeingVisited { depth: min_depth, annotation } => {
                WalkReturn::Cycle { min_depth, annotation }
            }

            NodeState::NotVisited => return None,

            NodeState::InCycleWith { parent } => panic!(
                "`find_state` returned `InCycleWith({parent:?})`, which ought to be impossible"
            ),
        })
    }

    /// Fetches the state of the node `r`. If `r` is recorded as being
    /// in a cycle with some other node `r2`, then fetches the state
    /// of `r2` (and updates `r` to reflect current result). This is
    /// basically the "find" part of a standard union-find algorithm
    /// (with path compression).
    fn find_state(&mut self, mut node: G::Node) -> NodeState<G::Node, A::SccIdx, A::Ann> {
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

        // Ultimately propagated to all the transitive parents when following
        // `InCycleWith` upwards.
        // This loop performs the downward link encoding mentioned above. Details below!
        // Note that there are two different states being assigned: the root state, and
        // a potentially derived version of the root state for non-root nodes in the chain.
        let (root_state, assigned_state) = {
            loop {
                trace!("find_state(r = {node:?} in state {:?})", self.node_states[node]);
                match self.node_states[node] {
                    // This must have been the first and only state since it is unexplored*;
                    // no update needed! * Unless there is a bug :')
                    s @ NodeState::NotVisited => return s,
                    // We are in a completely discovered SCC; every node on our path is in that SCC:
                    s @ NodeState::InCycle { .. } => break (s, s),
                    // The Interesting Third Base Case: we are a path back to a root node
                    // still being explored. Now we need that node to keep its state and
                    // every other node to be recorded as being in whatever component that
                    // ends up in.
                    s @ NodeState::BeingVisited { depth, .. } => {
                        break (s, NodeState::InCycleWith { parent: self.node_stack[depth] });
                    }
                    // We are not at the head of a path; keep compressing it!
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
            // Back at the beginning, we can return. Note that we return the root state.
            // This is because for components being explored, we would otherwise get a
            // `node_state[n] = InCycleWith{ parent: n }` and that's wrong.
            if previous_node == node {
                return root_state;
            }
            trace!("Compressing {node:?} down to {previous_node:?} with state {assigned_state:?}");

            // Update to previous node in the link.
            match self.node_states[previous_node] {
                NodeState::InCycleWith { parent: previous } => {
                    node = previous_node;
                    previous_node = previous;
                }
                // Only InCycleWith nodes were added to the reverse linked list.
                other => unreachable!("Invalid previous link while compressing cycle: {other:?}"),
            }

            // Update the node state to the (potentially derived) state.
            // If the root is still being explored, this is
            // `InCycleWith{ parent: <root node>}`, otherwise
            // `assigned_state == root_state`.
            self.node_states[node] = assigned_state;
        }
    }

    /// Walks a node that has never been visited before.
    ///
    /// Call this method when `inspect_node` has returned `None`. Having the
    /// caller decide avoids mutual recursion between the two methods and allows
    /// us to maintain an allocated stack for nodes on the path between calls.
    #[instrument(skip(self, initial), level = "trace")]
    fn walk_unvisited_node(&mut self, initial: G::Node) -> WalkReturn<A::SccIdx, A::Ann> {
        trace!("Walk unvisited node: {initial:?}");
        struct VisitingNodeFrame<G: DirectedGraph, Successors, A> {
            node: G::Node,
            successors: Option<Successors>,
            depth: usize,
            min_depth: usize,
            successors_len: usize,
            min_cycle_root: G::Node,
            successor_node: G::Node,
            /// The annotation for the SCC starting in `node`. It may or may
            /// not contain other nodes.
            current_component_annotation: A,
        }

        // Move the stack to a local variable. We want to utilize the existing allocation and
        // mutably borrow it without borrowing self at the same time.
        let mut successors_stack = core::mem::take(&mut self.successors_stack);

        debug_assert_eq!(successors_stack.len(), 0);

        let mut stack: Vec<VisitingNodeFrame<G, _, _>> = vec![VisitingNodeFrame {
            node: initial,
            depth: 0,
            min_depth: 0,
            successors: None,
            successors_len: 0,
            min_cycle_root: initial,
            successor_node: initial,
            current_component_annotation: self.annotations.new(initial),
        }];

        let mut return_value = None;

        'recurse: while let Some(frame) = stack.last_mut() {
            let VisitingNodeFrame {
                node,
                depth,
                successors,
                successors_len,
                min_depth,
                min_cycle_root,
                successor_node,
                current_component_annotation,
            } = frame;
            let node = *node;
            let depth = *depth;

            trace!(
                "Visiting {node:?} at depth {depth:?}, annotation: {current_component_annotation:?}"
            );

            let successors = match successors {
                Some(successors) => successors,
                None => {
                    // This None marks that we still have the initialize this node's frame.
                    trace!(?depth, ?node);

                    debug_assert_matches!(self.node_states[node], NodeState::NotVisited);

                    // Push `node` onto the stack.
                    self.node_states[node] = NodeState::BeingVisited {
                        depth,
                        annotation: *current_component_annotation,
                    };
                    self.node_stack.push(node);

                    // Walk each successor of the node, looking to see if any of
                    // them can reach a node that is presently on the stack. If
                    // so, that means they can also reach us.
                    *successors_len = successors_stack.len();
                    // Set and return a reference, this is currently empty.
                    successors.get_or_insert(self.graph.successors(node))
                }
            };

            // Now that the successors iterator is initialized, this is a constant for this frame.
            let successors_len = *successors_len;

            // Construct iterators for the nodes and walk results. There are two cases:
            // * The walk of a successor node returned.
            // * The remaining successor nodes.
            let returned_walk =
                return_value.take().into_iter().map(|walk| (*successor_node, Some(walk)));

            let successor_walk = successors.map(|successor_node| {
                trace!(?node, ?successor_node);
                (successor_node, self.inspect_node(successor_node))
            });
            for (successor_node, walk) in returned_walk.chain(successor_walk) {
                match walk {
                    // The starting node `node` leads to a cycle whose earliest node,
                    // `successor_node`, is at `min_depth`. There may be more cycles.
                    Some(WalkReturn::Cycle {
                        min_depth: successor_min_depth,
                        annotation: successor_annotation,
                    }) => {
                        trace!(
                            "Cycle found from {node:?}, minimum depth: {successor_min_depth:?}, annotation: {successor_annotation:?}"
                        );
                        // Track the minimum depth we can reach.
                        assert!(successor_min_depth <= depth);
                        if successor_min_depth < *min_depth {
                            trace!(?node, ?successor_min_depth);
                            *min_depth = successor_min_depth;
                            *min_cycle_root = successor_node;
                        }
                        current_component_annotation.update_scc(successor_annotation);
                    }
                    // The starting node `node` is succeeded by a fully identified SCC
                    // which is now added to the set under `scc_index`.
                    Some(WalkReturn::Complete {
                        scc_index: successor_scc_index,
                        annotation: successor_annotation,
                    }) => {
                        trace!(
                            "Complete; {node:?} is root of complete-visited SCC idx {successor_scc_index:?} with annotation {successor_annotation:?}"
                        );
                        // Push the completed SCC indices onto
                        // the `successors_stack` for later.
                        trace!(?node, ?successor_scc_index);
                        successors_stack.push(successor_scc_index);
                        current_component_annotation.update_reachable(successor_annotation);
                    }
                    // `node` has no more (direct) successors; search recursively.
                    None => {
                        let depth = depth + 1;
                        trace!("Recursing down into {successor_node:?} at depth {depth:?}");
                        trace!(?depth, ?successor_node);
                        // Remember which node the return value will come from.
                        frame.successor_node = successor_node;
                        // Start a new stack frame, then step into it.
                        stack.push(VisitingNodeFrame {
                            node: successor_node,
                            depth,
                            successors: None,
                            successors_len: 0,
                            min_depth: depth,
                            min_cycle_root: successor_node,
                            successor_node,
                            current_component_annotation: self.annotations.new(successor_node),
                        });
                        continue 'recurse;
                    }
                }
            }

            trace!("Finished walk from {node:?} with annotation: {current_component_annotation:?}");

            // Completed walk, remove `node` from the stack.
            let r = self.node_stack.pop();
            debug_assert_eq!(r, Some(node));

            // Remove the frame, it's done.
            let frame = stack.pop().unwrap();
            let current_component_annotation = frame.current_component_annotation;
            debug_assert_eq!(frame.node, node);

            // If `min_depth == depth`, then we are the root of the
            // cycle: we can't reach anyone further down the stack.

            // Pass the 'return value' down the stack.
            // We return one frame at a time so there can't be another return value.
            debug_assert!(return_value.is_none());
            return_value = Some(if frame.min_depth == depth {
                // We are at the head of the component.

                // Note that successor stack may have duplicates, so we
                // want to remove those:
                let deduplicated_successors = {
                    let duplicate_set = &mut self.duplicate_set;
                    duplicate_set.clear();
                    successors_stack
                        .drain(successors_len..)
                        .filter(move |&i| duplicate_set.insert(i))
                };

                debug!("Creating SCC rooted in {node:?} with successor {:?}", frame.successor_node);

                let scc_index = self.scc_data.create_scc(deduplicated_successors);

                self.annotations.annotate_scc(scc_index, current_component_annotation);

                self.node_states[node] =
                    NodeState::InCycle { scc_index, annotation: current_component_annotation };

                WalkReturn::Complete { scc_index, annotation: current_component_annotation }
            } else {
                // We are not the head of the cycle. Return back to our
                // caller. They will take ownership of the
                // `self.successors` data that we pushed.
                self.node_states[node] = NodeState::InCycleWith { parent: frame.min_cycle_root };
                WalkReturn::Cycle {
                    min_depth: frame.min_depth,
                    annotation: current_component_annotation,
                }
            });
        }

        // Keep the allocation we used for successors_stack.
        self.successors_stack = successors_stack;
        debug_assert_eq!(self.successors_stack.len(), 0);

        return_value.unwrap()
    }
}
