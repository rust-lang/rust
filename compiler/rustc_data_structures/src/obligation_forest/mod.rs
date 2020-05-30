//! matching to track the set of outstanding obligations (those not yet
//! resolved to success or error). It also tracks the "backtrace" of each
//! pending obligation (why we are trying to figure this out in the first
//! place).
//!
//! ### External view
//!
//! `ObligationForest` supports two main public operations (there are a
//! few others not discussed here):
//!
//! 1. Add a new root obligations (`register_obligation`).
//! 2. Process the pending obligations (`process_obligations`).
//!
//! When a new obligation `N` is added, it becomes the root of an
//! obligation tree. This tree can also carry some per-tree state `T`,
//! which is given at the same time. This tree is a singleton to start, so
//! `N` is both the root and the only leaf. Each time the
//! `process_obligations` method is called, it will invoke its callback
//! with every pending obligation (so that will include `N`, the first
//! time). The callback also receives a (mutable) reference to the
//! per-tree state `T`. The callback should process the obligation `O`
//! that it is given and return a `ProcessResult`:
//!
//! - `Unchanged` -> ambiguous result. Obligation was neither a success
//!   nor a failure. It is assumed that further attempts to process the
//!   obligation will yield the same result unless something in the
//!   surrounding environment changes.
//! - `Changed(C)` - the obligation was *shallowly successful*. The
//!   vector `C` is a list of subobligations. The meaning of this is that
//!   `O` was successful on the assumption that all the obligations in `C`
//!   are also successful. Therefore, `O` is only considered a "true"
//!   success if `C` is empty. Otherwise, `O` is put into a suspended
//!   state and the obligations in `C` become the new pending
//!   obligations. They will be processed the next time you call
//!   `process_obligations`.
//! - `Error(E)` -> obligation failed with error `E`. We will collect this
//!   error and return it from `process_obligations`, along with the
//!   "backtrace" of obligations (that is, the list of obligations up to
//!   and including the root of the failed obligation). No further
//!   obligations from that same tree will be processed, since the tree is
//!   now considered to be in error.
//!
//! When the call to `process_obligations` completes, you get back an `Outcome`,
//! which includes three bits of information:
//!
//! - `completed`: a list of obligations where processing was fully
//!   completed without error (meaning that all transitive subobligations
//!   have also been completed). So, for example, if the callback from
//!   `process_obligations` returns `Changed(C)` for some obligation `O`,
//!   then `O` will be considered completed right away if `C` is the
//!   empty vector. Otherwise it will only be considered completed once
//!   all the obligations in `C` have been found completed.
//! - `errors`: a list of errors that occurred and associated backtraces
//!   at the time of error, which can be used to give context to the user.
//! - `stalled`: if true, then none of the existing obligations were
//!   *shallowly successful* (that is, no callback returned `Changed(_)`).
//!   This implies that all obligations were either errors or returned an
//!   ambiguous result, which means that any further calls to
//!   `process_obligations` would simply yield back further ambiguous
//!   results. This is used by the `FulfillmentContext` to decide when it
//!   has reached a steady state.
//!
//! ### Implementation details
//!
//! For the most part, comments specific to the implementation are in the
//! code. This file only contains a very high-level overview. Basically,
//! the forest is stored in a vector. Each element of the vector is a node
//! in some tree. Each node in the vector has the index of its dependents,
//! including the first dependent which is known as the parent. It also
//! has a current state, described by `NodeState`. After each processing
//! step, we compress the vector to remove completed and error nodes, which
//! aren't needed anymore.

use crate::fx::{FxHashMap, FxHashSet};

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::hash;
use std::marker::PhantomData;
use std::mem;

mod graphviz;

#[cfg(test)]
mod tests;

pub trait ForestObligation: Clone + Debug {
    /// A key used to avoid evaluating the same obligation twice
    type CacheKey: Clone + hash::Hash + Eq + Debug;
    /// The variable type used in the obligation when it could not yet be fulfilled
    type Variable: Clone + hash::Hash + Eq + Debug;
    /// A type which tracks which variables has been unified
    type WatcherOffset;

    /// Converts this `ForestObligation` suitable for use as a cache key.
    /// If two distinct `ForestObligations`s return the same cache key,
    /// then it must be sound to use the result of processing one obligation
    /// (e.g. success for error) for the other obligation
    fn as_cache_key(&self) -> Self::CacheKey;

    /// Returns which variables this obligation is currently stalled on. If the slice is empty then
    /// the variables stalled on are unknown.
    fn stalled_on(&self) -> &[Self::Variable];
}

pub trait ObligationProcessor {
    type Obligation: ForestObligation;
    type Error: Debug;

    fn process_obligation(
        &mut self,
        obligation: &mut Self::Obligation,
    ) -> ProcessResult<Self::Obligation, Self::Error>;

    /// As we do the cycle check, we invoke this callback when we
    /// encounter an actual cycle. `cycle` is an iterator that starts
    /// at the start of the cycle in the stack and walks **toward the
    /// top**.
    ///
    /// In other words, if we had O1 which required O2 which required
    /// O3 which required O1, we would give an iterator yielding O1,
    /// O2, O3 (O1 is not yielded twice).
    fn process_backedge<'c, I>(&mut self, cycle: I, _marker: PhantomData<&'c Self::Obligation>)
    where
        I: Clone + Iterator<Item = &'c Self::Obligation>;

    fn unblocked(
        &self,
        offset: &<Self::Obligation as ForestObligation>::WatcherOffset,
        f: impl FnMut(<Self::Obligation as ForestObligation>::Variable),
    );
    fn register_variable_watcher(&self) -> <Self::Obligation as ForestObligation>::WatcherOffset;
    fn deregister_variable_watcher(
        &self,
        offset: <Self::Obligation as ForestObligation>::WatcherOffset,
    );
    fn watch_variable(&self, var: <Self::Obligation as ForestObligation>::Variable);
    fn unwatch_variable(&self, var: <Self::Obligation as ForestObligation>::Variable);
}

/// The result type used by `process_obligation`.
#[derive(Debug)]
pub enum ProcessResult<O, E> {
    Unchanged,
    Changed(Vec<O>),
    Error(E),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ObligationTreeId(usize);

type ObligationTreeIdGenerator =
    std::iter::Map<std::ops::RangeFrom<usize>, fn(usize) -> ObligationTreeId>;

/// `usize` indices are used here and throughout this module, rather than
/// `rustc_index::newtype_index!` indices, because this code is hot enough
/// that the `u32`-to-`usize` conversions that would be required are
/// significant, and space considerations are not important.
type NodeIndex = usize;

enum CacheState {
    Active(NodeIndex),
    Done,
}

pub struct ObligationForest<O: ForestObligation> {
    /// The list of obligations. In between calls to `process_obligations`,
    /// this list only contains nodes in the `Pending` or `Waiting` state.
    nodes: Vec<Node<O>>,

    /// Nodes must be processed in the order that they were added so we give each node a unique,
    /// number allowing them to be ordered when processing them.
    node_number: u32,

    /// Stores the indices of the nodes currently in the pending state
    pending_nodes: Vec<NodeIndex>,

    /// Stores the indices of the nodes currently in the success or waiting states.
    /// Can also contain `Done` or `Error` nodes as `process_cycles` does not remove a node
    /// immediately but instead upon the next time that node is processed.
    success_or_waiting_nodes: Vec<NodeIndex>,
    /// Stores the indices of the nodes currently in the error or done states
    error_or_done_nodes: RefCell<Vec<NodeIndex>>,

    /// Nodes that have been removed and are ready to be reused (pure optimization to reuse
    /// allocations)
    dead_nodes: Vec<NodeIndex>,

    /// A cache of the nodes in `nodes`, indexed by predicate. Unfortunately,
    /// its contents are not guaranteed to match those of `nodes`. See the
    /// comments in `process_obligation` for details.
    active_cache: FxHashMap<O::CacheKey, CacheState>,

    obligation_tree_id_generator: ObligationTreeIdGenerator,

    /// Per tree error cache. This is used to deduplicate errors,
    /// which is necessary to avoid trait resolution overflow in
    /// some cases.
    ///
    /// See [this][details] for details.
    ///
    /// [details]: https://github.com/rust-lang/rust/pull/53255#issuecomment-421184780
    error_cache: FxHashMap<ObligationTreeId, FxHashSet<O::CacheKey>>,

    /// Stores which nodes would be unblocked once `O::Variable` is unified
    stalled_on: FxHashMap<O::Variable, Vec<NodeIndex>>,
    /// Stores the node indices that are unblocked and should be processed at the next opportunity
    unblocked: BinaryHeap<Unblocked>,
    /// Stores nodes which should be processed on the next iteration since the variables they are
    /// actually blocked on are unknown
    stalled_on_unknown: Vec<NodeIndex>,
    /// The offset that this `ObligationForest` has registered. Should be de-registered before
    /// dropping this forest.
    offset: Option<O::WatcherOffset>,
    /// Reusable vector for storing unblocked nodes whose watch should be removed
    temp_unblocked_nodes: Vec<O::Variable>,
}

/// Helper struct for use with `BinaryHeap` to process nodes in the order that they were added to
/// the forest
struct Unblocked {
    index: NodeIndex,
    order: u32,
}

impl PartialEq for Unblocked {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}
impl Eq for Unblocked {}
impl PartialOrd for Unblocked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.order.partial_cmp(&self.order)
    }
}
impl Ord for Unblocked {
    fn cmp(&self, other: &Self) -> Ordering {
        other.order.cmp(&self.order)
    }
}

#[derive(Debug)]
struct Node<O: ForestObligation> {
    obligation: O,
    state: Cell<NodeState>,

    /// A predicate (and its key) can changed during processing. If it does we need to register the
    /// old predicate so that we can remove or mark it as done if this node errors or is done.
    alternative_predicates: Vec<O::CacheKey>,

    /// Obligations that depend on this obligation for their completion. They
    /// must all be in a non-pending state.
    dependents: Vec<NodeIndex>,
    /// Obligations that this obligation depends on for their completion.
    reverse_dependents: Vec<NodeIndex>,

    /// If true, `dependents[0]` points to a "parent" node, which requires
    /// special treatment upon error but is otherwise treated the same.
    /// (It would be more idiomatic to store the parent node in a separate
    /// `Option<NodeIndex>` field, but that slows down the common case of
    /// iterating over the parent and other descendants together.)
    has_parent: bool,

    /// Identifier of the obligation tree to which this node belongs.
    obligation_tree_id: ObligationTreeId,
    node_number: u32,
}

impl<O> Node<O>
where
    O: ForestObligation,
{
    fn new(
        parent: Option<NodeIndex>,
        obligation: O,
        obligation_tree_id: ObligationTreeId,
        node_number: u32,
    ) -> Node<O> {
        Node {
            obligation,
            state: Cell::new(NodeState::Pending),
            alternative_predicates: vec![],
            dependents: if let Some(parent_index) = parent { vec![parent_index] } else { vec![] },
            reverse_dependents: vec![],
            has_parent: parent.is_some(),
            obligation_tree_id,
            node_number,
        }
    }

    /// Initializes a node, reusing the existing allocations
    fn init(
        &mut self,
        parent: Option<NodeIndex>,
        obligation: O,
        obligation_tree_id: ObligationTreeId,
        node_number: u32,
    ) {
        self.obligation = obligation;
        debug_assert!(
            self.state.get() == NodeState::Done || self.state.get() == NodeState::Error,
            "{:?}",
            self.state
        );
        self.state.set(NodeState::Pending);
        self.alternative_predicates.clear();
        self.dependents.clear();
        self.reverse_dependents.clear();
        if let Some(parent_index) = parent {
            self.dependents.push(parent_index);
        }
        self.has_parent = parent.is_some();
        self.obligation_tree_id = obligation_tree_id;
        self.node_number = node_number;
    }
}

/// The state of one node in some tree within the forest. This represents the
/// current state of processing for the obligation (of type `O`) associated
/// with this node.
///
/// The non-`Error` state transitions are as follows.
/// ```
/// (Pre-creation)
///  |
///  |     register_obligation_at() (called by process_obligations() and
///  v                               from outside the crate)
/// Pending
///  |
///  |     process_obligations()
///  v
/// Success
///  |  ^
///  |  |  mark_successes()
///  |  v
///  |  Waiting
///  |
///  |     process_cycles()
///  v
/// Done
///  |
///  |     compress()
///  v
/// (Removed)
/// ```
/// The `Error` state can be introduced in several places, via `error_at()`.
///
/// Outside of `ObligationForest` methods, nodes should be either `Pending` or
/// `Waiting`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NodeState {
    /// This obligation has not yet been selected successfully. Cannot have
    /// subobligations.
    Pending,

    /// This obligation was selected successfully, but may or may not have
    /// subobligations.
    Success,

    /// This obligation was selected successfully, but it has a pending
    /// subobligation.
    Waiting,

    /// This obligation, along with its subobligations, are complete, and will
    /// be removed in the next collection.
    Done,

    /// This obligation was resolved to an error. It will be removed by the
    /// next compression step.
    Error,
}

/// This trait allows us to have two different Outcome types:
///  - the normal one that does as little as possible
///  - one for tests that does some additional work and checking
pub trait OutcomeTrait {
    type Error;
    type Obligation;

    fn new() -> Self;
    fn mark_not_stalled(&mut self);
    fn is_stalled(&self) -> bool;
    fn record_completed(&mut self, outcome: &Self::Obligation);
    fn record_error(&mut self, error: Self::Error);
}

#[derive(Debug)]
pub struct Outcome<O, E> {
    /// Backtrace of obligations that were found to be in error.
    pub errors: Vec<Error<O, E>>,
}

impl<O, E> OutcomeTrait for Outcome<O, E> {
    type Error = Error<O, E>;
    type Obligation = O;

    fn new() -> Self {
        Self { stalled: true, errors: vec![] }
    }

    fn mark_not_stalled(&mut self) {
        self.stalled = false;
    }

    fn is_stalled(&self) -> bool {
        self.stalled
    }

    fn record_completed(&mut self, _outcome: &Self::Obligation) {
        // do nothing
    }

    fn record_error(&mut self, error: Self::Error) {
        self.errors.push(error)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Error<O, E> {
    pub error: E,
    pub backtrace: Vec<O>,
}

impl<O: ForestObligation> ObligationForest<O> {
    pub fn new() -> ObligationForest<O> {
        ObligationForest {
            nodes: vec![],
            pending_nodes: vec![],
            success_or_waiting_nodes: vec![],
            error_or_done_nodes: RefCell::new(vec![]),
            dead_nodes: vec![],
            active_cache: Default::default(),
            obligation_tree_id_generator: (0..).map(ObligationTreeId),
            node_number: 0,
            error_cache: Default::default(),
            stalled_on: Default::default(),
            unblocked: Default::default(),
            stalled_on_unknown: Default::default(),
            temp_unblocked_nodes: Default::default(),
            offset: None,
        }
    }

    pub fn offset(&self) -> Option<&O::WatcherOffset> {
        self.offset.as_ref()
    }

    pub fn take_offset(&mut self) -> Option<O::WatcherOffset> {
        self.offset.take()
    }

    /// Returns the total number of nodes in the forest that have not
    /// yet been fully resolved.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Registers an obligation.
    pub fn register_obligation(&mut self, obligation: O) {
        // Ignore errors here - there is no guarantee of success.
        let _ = self.register_obligation_at(obligation, None);
    }

    // Returns Err(()) if we already know this obligation failed.
    fn register_obligation_at(
        &mut self,
        obligation: O,
        parent: Option<NodeIndex>,
    ) -> Result<(), ()> {
        debug_assert!(obligation.stalled_on().is_empty());
        match self.active_cache.entry(obligation.as_cache_key().clone()) {
            Entry::Occupied(o) => {
                let index = match o.get() {
                    CacheState::Active(index) => *index,
                    CacheState::Done => {
                        debug!(
                            "register_obligation_at: ignoring already done obligation: {:?}",
                            obligation
                        );
                        return Ok(());
                    }
                };
                let node = &mut self.nodes[index];
                let state = node.state.get();
                if let Some(parent_index) = parent {
                    // If the node is already in `active_cache`, it has already
                    // had its chance to be marked with a parent. So if it's
                    // not already present, just dump `parent` into the
                    // dependents as a non-parent.
                    if !node.dependents.contains(&parent_index) {
                        node.dependents.push(parent_index);
                        self.nodes[parent_index].reverse_dependents.push(index);
                    }
                }
                if let NodeState::Error = state { Err(()) } else { Ok(()) }
            }
            Entry::Vacant(v) => {
                let obligation_tree_id = match parent {
                    Some(parent_index) => self.nodes[parent_index].obligation_tree_id,
                    None => self.obligation_tree_id_generator.next().unwrap(),
                };

                let already_failed = parent.is_some()
                    && self
                        .error_cache
                        .get(&obligation_tree_id)
                        .map(|errors| errors.contains(&obligation.as_cache_key()))
                        .unwrap_or(false);

                if already_failed {
                    Err(())
                } else {
                    // Retrieves a fresh number for the new node so that each node are processed in the
                    // order that they were created
                    let node_number = self.node_number;
                    self.node_number += 1;

                    // If we have a dead node we can reuse it and it's associated allocations,
                    // otherwise allocate a new node
                    let new_index = if let Some(new_index) = self.dead_nodes.pop() {
                        let node = &mut self.nodes[new_index];
                        node.init(parent, obligation, obligation_tree_id, node_number);
                        new_index
                    } else {
                        let new_index = self.nodes.len();
                        self.nodes.push(Node::new(
                            parent,
                            obligation,
                            obligation_tree_id,
                            node_number,
                        ));
                        new_index
                    };
                    if let Some(parent_index) = parent {
                        self.nodes[parent_index].reverse_dependents.push(new_index);
                    }

                    self.pending_nodes.push(new_index);
                    self.unblocked.push(Unblocked { index: new_index, order: node_number });
                    v.insert(CacheState::Active(new_index));
                    Ok(())
                }
            }
        }
    }

    /// Converts all remaining obligations to the given error.
    pub fn to_errors<E: Clone>(&mut self, error: E) -> Vec<Error<O, E>> {
        let errors = self
            .pending_nodes
            .iter()
            .map(|&index| Error { error: error.clone(), backtrace: self.error_at(index) })
            .collect();

        self.compress(|_| assert!(false));
        errors
    }

    /// Returns the set of obligations that are in a pending state.
    pub fn map_pending_obligations<P, F>(&self, f: F) -> Vec<P>
    where
        F: Fn(&O) -> P,
    {
        self.pending_nodes
            .iter()
            .map(|&index| &self.nodes[index])
            .map(|node| f(&node.obligation))
            .collect()
    }

    fn insert_into_error_cache(&mut self, index: NodeIndex) {
        let node = &self.nodes[index];
        self.error_cache
            .entry(node.obligation_tree_id)
            .or_default()
            .insert(node.obligation.as_cache_key());
    }

    /// Performs a pass through the obligation list. This must
    /// be called in a loop until `outcome.stalled` is false.
    ///
    /// This _cannot_ be unrolled (presently, at least).
    pub fn process_obligations<P, OUT>(&mut self, processor: &mut P) -> OUT
    where
        P: ObligationProcessor<Obligation = O>,
        OUT: OutcomeTrait<Obligation = O, Error = Error<O, P::Error>>,
    {
        if self.offset.is_none() {
            self.offset = Some(processor.register_variable_watcher());
        }
        let mut errors = vec![];
        let mut stalled = true;

        self.unblock_nodes(processor);

        let nodes = &self.nodes;
        self.unblocked.extend(
            self.stalled_on_unknown
                .drain(..)
                .map(|index| Unblocked { index, order: nodes[index].node_number }),
        );
        while let Some(Unblocked { index, .. }) = self.unblocked.pop() {
            // Skip any duplicates since we only need to processes the node once
            if self.unblocked.peek().map(|u| u.index) == Some(index) {
                continue;
            }

            let node = &mut self.nodes[index];

            if node.state.get() != NodeState::Pending {
                continue;
            }

            // `processor.process_obligation` can modify the predicate within
            // `node.obligation`, and that predicate is the key used for
            // `self.active_cache`. This means that `self.active_cache` can get
            // out of sync with `nodes`. It's not very common, but it does
            // happen, and code in `compress` has to allow for it.
            let before = node.obligation.as_cache_key();
            let result = processor.process_obligation(&mut node.obligation);
            let after = node.obligation.as_cache_key();
            if before != after {
                node.alternative_predicates.push(before);
            }

            self.unblock_nodes(processor);
            let node = &mut self.nodes[index];
            match result {
                ProcessResult::Unchanged => {
                    let stalled_on = node.obligation.stalled_on();
                    if stalled_on.is_empty() {
                        // We stalled but the variables that caused it are unknown so we run
                        // `index` again at the next opportunity
                        self.stalled_on_unknown.push(index);
                    } else {
                        // Register every variable that we stalled on
                        for var in stalled_on {
                            self.stalled_on
                                .entry(var.clone())
                                .or_insert_with(|| {
                                    processor.watch_variable(var.clone());
                                    Vec::new()
                                })
                                .push(index);
                        }
                    }
                    // No change in state.
                }
                ProcessResult::Changed(children) => {
                    // We are not (yet) stalled.
                    stalled = false;
                    node.state.set(NodeState::Success);
                    self.success_or_waiting_nodes.push(index);

                    for child in children {
                        let st = self.register_obligation_at(child, Some(index));
                        if let Err(()) = st {
                            // Error already reported - propagate it
                            // to our node.
                            self.error_at(index);
                        }
                    }
                }
                ProcessResult::Error(err) => {
                    stalled = false;
                    errors.push(Error { error: err, backtrace: self.error_at(index) });
                }
            }
        }

        if stalled {
            // There's no need to perform marking, cycle processing and compression when nothing
            // changed.
            return Outcome {
                completed: if do_completed == DoCompleted::Yes { Some(vec![]) } else { None },
                errors,
            };
        }

        self.mark_successes();
        self.process_cycles(processor);
        let completed = self.compress(do_completed);

        Outcome { completed, errors }
    }

    #[inline(never)]
    fn unblock_nodes<P>(&mut self, processor: &mut P)
    where
        P: ObligationProcessor<Obligation = O>,
    {
        let nodes = &mut self.nodes;
        let stalled_on = &mut self.stalled_on;
        let unblocked = &mut self.unblocked;
        let temp_unblocked_nodes = &mut self.temp_unblocked_nodes;
        temp_unblocked_nodes.clear();
        processor.unblocked(self.offset.as_ref().unwrap(), |var| {
            if let Some(unblocked_nodes) = stalled_on.remove(&var) {
                for node_index in unblocked_nodes {
                    let node = &nodes[node_index];
                    debug_assert!(
                        node.state.get() == NodeState::Pending,
                        "Unblocking non-pending2: {:?}",
                        node.obligation
                    );
                    unblocked.push(Unblocked { index: node_index, order: node.node_number });
                }
                temp_unblocked_nodes.push(var);
            }
        });
        for var in temp_unblocked_nodes.drain(..) {
            processor.unwatch_variable(var);
        }
    }

    /// Returns a vector of obligations for `p` and all of its
    /// ancestors, putting them into the error state in the process.
    fn error_at(&self, mut index: NodeIndex) -> Vec<O> {
        let mut error_stack: Vec<NodeIndex> = vec![];
        let mut trace = vec![];

        let mut error_or_done_nodes = self.error_or_done_nodes.borrow_mut();

        loop {
            let node = &self.nodes[index];
            match node.state.get() {
                NodeState::Error | NodeState::Done => (), // Already added to `error_or_done_nodes`
                _ => error_or_done_nodes.push(index),
            }
            node.state.set(NodeState::Error);
            trace.push(node.obligation.clone());
            if node.has_parent {
                // The first dependent is the parent, which is treated
                // specially.
                error_stack.extend(node.dependents.iter().skip(1));
                index = node.dependents[0];
            } else {
                // No parent; treat all dependents non-specially.
                error_stack.extend(node.dependents.iter());
                break;
            }
        }

        while let Some(index) = error_stack.pop() {
            let node = &self.nodes[index];
            if node.state.get() != NodeState::Error {
                node.state.set(NodeState::Error);
                error_stack.extend(node.dependents.iter());
            }
        }

        trace
    }

    /// Mark all `Waiting` nodes as `Success`, except those that depend on a
    /// pending node.
    fn mark_successes(&mut self) {
        // Convert all `Waiting` nodes to `Success`.
        for &index in &self.success_or_waiting_nodes {
            let node = &self.nodes[index];
            if node.state.get() == NodeState::Waiting {
                node.state.set(NodeState::Success);
            }
        }

        // Convert `Success` nodes that depend on a pending node back to
        // `Waiting`.
        let mut pending_nodes = mem::take(&mut self.pending_nodes);
        pending_nodes.retain(|&index| {
            let node = &self.nodes[index];
            if node.state.get() == NodeState::Pending {
                // This call site is hot.
                self.inlined_mark_dependents_as_waiting(node);
                true
            } else {
                false
            }
        });
        self.pending_nodes = pending_nodes;
    }

    // This always-inlined function is for the hot call site.
    #[inline(always)]
    fn inlined_mark_dependents_as_waiting(&self, node: &Node<O>) {
        for &index in node.dependents.iter() {
            let node = &self.nodes[index];
            let state = node.state.get();
            if state == NodeState::Success {
                // This call site is cold.
                self.uninlined_mark_dependents_as_waiting(node);
            } else {
                debug_assert!(state == NodeState::Waiting || state == NodeState::Error)
            }
        }
    }

    // This never-inlined function is for the cold call site.
    #[inline(never)]
    fn uninlined_mark_dependents_as_waiting(&self, node: &Node<O>) {
        // Mark node Waiting in the cold uninlined code instead of the hot inlined
        node.state.set(NodeState::Waiting);
        self.inlined_mark_dependents_as_waiting(node)
    }

    /// Report cycles between all `Success` nodes, and convert all `Success`
    /// nodes to `Done`. This must be called after `mark_successes`.
    fn process_cycles<P>(&mut self, processor: &mut P)
    where
        P: ObligationProcessor<Obligation = O>,
    {
        let mut stack = vec![];

        let success_or_waiting_nodes = mem::take(&mut self.success_or_waiting_nodes);
        for &index in &success_or_waiting_nodes {
            let node = &self.nodes[index];
            // For some benchmarks this state test is extremely hot. It's a win
            // to handle the no-op cases immediately to avoid the cost of the
            // function call.
            if node.state.get() == NodeState::Success {
                self.find_cycles_from_node(&mut stack, processor, index);
            }
        }
        self.success_or_waiting_nodes = success_or_waiting_nodes;

        debug_assert!(stack.is_empty());
        self.reused_node_vec = stack;
    }

    fn find_cycles_from_node<P>(
        &self,
        stack: &mut Vec<NodeIndex>,
        processor: &mut P,
        index: NodeIndex,
    ) where
        P: ObligationProcessor<Obligation = O>,
    {
        let node = &self.nodes[index];
        if node.state.get() == NodeState::Success {
            match stack.iter().rposition(|&n| n == index) {
                None => {
                    stack.push(index);
                    for &dep_index in node.dependents.iter() {
                        self.find_cycles_from_node(stack, processor, dep_index);
                    }
                    stack.pop();
                    node.state.set(NodeState::Done);
                    self.error_or_done_nodes.borrow_mut().push(index);
                }
                Some(rpos) => {
                    // Cycle detected.
                    processor.process_backedge(
                        stack[rpos..].iter().map(|i| &self.nodes[*i].obligation),
                        PhantomData,
                    );
                }
            }
        }
    }

    /// Compresses the vector, removing all popped nodes. This adjusts the
    /// indices and hence invalidates any outstanding indices. `process_cycles`
    /// must be run beforehand to remove any cycles on `Success` nodes.
    #[inline(never)]
    fn compress(&mut self, do_completed: DoCompleted) -> Option<Vec<O>> {
        let mut removed_done_obligations: Vec<O> = vec![];

        // Compress the forest by removing any nodes marked as error or done
        let mut error_or_done_nodes = mem::take(self.error_or_done_nodes.get_mut());
        for &index in &error_or_done_nodes {
            let node = &mut self.nodes[index];
            let reverse_dependents = mem::take(&mut node.reverse_dependents);
            for &reverse_index in &reverse_dependents {
                let reverse_node = &mut self.nodes[reverse_index];

                if let Some(i) = reverse_node.dependents.iter().position(|x| *x == index) {
                    reverse_node.dependents.swap_remove(i);
                    if i == 0 {
                        reverse_node.has_parent = false;
                    }
                }
            }
            let node = &mut self.nodes[index];
            node.reverse_dependents = reverse_dependents;

            match node.state.get() {
                NodeState::Done => {
                    // Mark as done
                    if let Some(opt) = self.active_cache.get_mut(&node.obligation.as_cache_key()) {
                        *opt = CacheState::Done;
                    }
                    for alt in &node.alternative_predicates {
                        if let Some(opt) = self.active_cache.get_mut(alt) {
                            *opt = CacheState::Done;
                        }
                    }

                    if do_completed == DoCompleted::Yes {
                        // Extract the success stories.
                        removed_done_obligations.push(node.obligation.clone());
                    }

                    self.dead_nodes.push(index);
                }
                NodeState::Error => {
                    // We *intentionally* remove the node from the cache at this point. Otherwise
                    // tests must come up with a different type on every type error they
                    // check against.
                    self.active_cache.remove(&node.obligation.as_cache_key());
                    for alt in &node.alternative_predicates {
                        self.active_cache.remove(alt);
                    }
                    self.insert_into_error_cache(index);
                    self.dead_nodes.push(index);
                }
                NodeState::Pending | NodeState::Waiting | NodeState::Success => unreachable!(),
            }
        }
        error_or_done_nodes.clear();
        *self.error_or_done_nodes.get_mut() = error_or_done_nodes;

        let nodes = &self.nodes;
        self.success_or_waiting_nodes.retain(|&index| match nodes[index].state.get() {
            NodeState::Waiting | NodeState::Success => true,
            NodeState::Done | NodeState::Error => false,
            NodeState::Pending => unreachable!(),
        });

        if do_completed == DoCompleted::Yes { Some(removed_done_obligations) } else { None }
    }
}
