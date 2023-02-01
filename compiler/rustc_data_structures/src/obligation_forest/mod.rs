//! The `ObligationForest` is a utility data structure used in trait
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
//! which includes two bits of information:
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
//!
//! Upon completion, none of the existing obligations were *shallowly
//! successful* (that is, no callback returned `Changed(_)`). This implies that
//! all obligations were either errors or returned an ambiguous result.
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

use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash;
use std::marker::PhantomData;

mod graphviz;

#[cfg(test)]
mod tests;

pub trait ForestObligation: Clone + Debug {
    type CacheKey: Clone + hash::Hash + Eq + Debug;

    /// Converts this `ForestObligation` suitable for use as a cache key.
    /// If two distinct `ForestObligations`s return the same cache key,
    /// then it must be sound to use the result of processing one obligation
    /// (e.g. success for error) for the other obligation
    fn as_cache_key(&self) -> Self::CacheKey;
}

pub trait ObligationProcessor {
    type Obligation: ForestObligation;
    type Error: Debug;
    type OUT: OutcomeTrait<Obligation = Self::Obligation, Error = Error<Self::Obligation, Self::Error>>;

    fn needs_process_obligation(&self, obligation: &Self::Obligation) -> bool;

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
    fn process_backedge<'c, I>(
        &mut self,
        cycle: I,
        _marker: PhantomData<&'c Self::Obligation>,
    ) -> Result<(), Self::Error>
    where
        I: Clone + Iterator<Item = &'c Self::Obligation>;
}

/// The result type used by `process_obligation`.
// `repr(C)` to inhibit the niche filling optimization. Otherwise, the `match` appearing
// in `process_obligations` is significantly slower, which can substantially affect
// benchmarks like `rustc-perf`'s inflate and keccak.
#[repr(C)]
#[derive(Debug)]
pub enum ProcessResult<O, E> {
    Unchanged,
    Changed(Vec<O>),
    Error(E),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ObligationTreeId(usize);

type ObligationTreeIdGenerator = impl Iterator<Item = ObligationTreeId>;

pub struct ObligationForest<O: ForestObligation> {
    /// The list of obligations. In between calls to [Self::process_obligations],
    /// this list only contains nodes in the `Pending` or `Waiting` state.
    ///
    /// `usize` indices are used here and throughout this module, rather than
    /// [`rustc_index::newtype_index!`] indices, because this code is hot enough
    /// that the `u32`-to-`usize` conversions that would be required are
    /// significant, and space considerations are not important.
    nodes: Vec<Node<O>>,

    /// A cache of predicates that have been successfully completed.
    done_cache: FxHashSet<O::CacheKey>,

    /// A cache of the nodes in `nodes`, indexed by predicate. Unfortunately,
    /// its contents are not guaranteed to match those of `nodes`. See the
    /// comments in `Self::process_obligation` for details.
    active_cache: FxHashMap<O::CacheKey, usize>,

    /// A vector reused in [Self::compress()] and [Self::find_cycles_from_node()],
    /// to avoid allocating new vectors.
    reused_node_vec: Vec<usize>,

    obligation_tree_id_generator: ObligationTreeIdGenerator,

    /// Per tree error cache. This is used to deduplicate errors,
    /// which is necessary to avoid trait resolution overflow in
    /// some cases.
    ///
    /// See [this][details] for details.
    ///
    /// [details]: https://github.com/rust-lang/rust/pull/53255#issuecomment-421184780
    error_cache: FxHashMap<ObligationTreeId, FxHashSet<O::CacheKey>>,
}

#[derive(Debug)]
struct Node<O> {
    obligation: O,
    state: Cell<NodeState>,

    /// Obligations that depend on this obligation for their completion. They
    /// must all be in a non-pending state.
    dependents: Vec<usize>,

    /// If true, `dependents[0]` points to a "parent" node, which requires
    /// special treatment upon error but is otherwise treated the same.
    /// (It would be more idiomatic to store the parent node in a separate
    /// `Option<usize>` field, but that slows down the common case of
    /// iterating over the parent and other descendants together.)
    has_parent: bool,

    /// Identifier of the obligation tree to which this node belongs.
    obligation_tree_id: ObligationTreeId,
}

impl<O> Node<O> {
    fn new(parent: Option<usize>, obligation: O, obligation_tree_id: ObligationTreeId) -> Node<O> {
        Node {
            obligation,
            state: Cell::new(NodeState::Pending),
            dependents: if let Some(parent_index) = parent { vec![parent_index] } else { vec![] },
            has_parent: parent.is_some(),
            obligation_tree_id,
        }
    }
}

/// The state of one node in some tree within the forest. This represents the
/// current state of processing for the obligation (of type `O`) associated
/// with this node.
///
/// The non-`Error` state transitions are as follows.
/// ```text
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
        Self { errors: vec![] }
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
            done_cache: Default::default(),
            active_cache: Default::default(),
            reused_node_vec: vec![],
            obligation_tree_id_generator: (0..).map(ObligationTreeId),
            error_cache: Default::default(),
        }
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
    fn register_obligation_at(&mut self, obligation: O, parent: Option<usize>) -> Result<(), ()> {
        let cache_key = obligation.as_cache_key();
        if self.done_cache.contains(&cache_key) {
            debug!("register_obligation_at: ignoring already done obligation: {:?}", obligation);
            return Ok(());
        }

        match self.active_cache.entry(cache_key) {
            Entry::Occupied(o) => {
                let node = &mut self.nodes[*o.get()];
                if let Some(parent_index) = parent {
                    // If the node is already in `active_cache`, it has already
                    // had its chance to be marked with a parent. So if it's
                    // not already present, just dump `parent` into the
                    // dependents as a non-parent.
                    if !node.dependents.contains(&parent_index) {
                        node.dependents.push(parent_index);
                    }
                }
                if let NodeState::Error = node.state.get() { Err(()) } else { Ok(()) }
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
                        .map_or(false, |errors| errors.contains(v.key()));

                if already_failed {
                    Err(())
                } else {
                    let new_index = self.nodes.len();
                    v.insert(new_index);
                    self.nodes.push(Node::new(parent, obligation, obligation_tree_id));
                    Ok(())
                }
            }
        }
    }

    /// Converts all remaining obligations to the given error.
    pub fn to_errors<E: Clone>(&mut self, error: E) -> Vec<Error<O, E>> {
        let errors = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_index, node)| node.state.get() == NodeState::Pending)
            .map(|(index, _node)| Error { error: error.clone(), backtrace: self.error_at(index) })
            .collect();

        self.compress(|_| assert!(false));
        errors
    }

    /// Returns the set of obligations that are in a pending state.
    pub fn map_pending_obligations<P, F>(&self, f: F) -> Vec<P>
    where
        F: Fn(&O) -> P,
    {
        self.nodes
            .iter()
            .filter(|node| node.state.get() == NodeState::Pending)
            .map(|node| f(&node.obligation))
            .collect()
    }

    fn insert_into_error_cache(&mut self, index: usize) {
        let node = &self.nodes[index];
        self.error_cache
            .entry(node.obligation_tree_id)
            .or_default()
            .insert(node.obligation.as_cache_key());
    }

    /// Performs a fixpoint computation over the obligation list.
    #[inline(never)]
    pub fn process_obligations<P>(&mut self, processor: &mut P) -> P::OUT
    where
        P: ObligationProcessor<Obligation = O>,
    {
        let mut outcome = P::OUT::new();

        // Fixpoint computation: we repeat until the inner loop stalls.
        loop {
            let mut has_changed = false;

            // Note that the loop body can append new nodes, and those new nodes
            // will then be processed by subsequent iterations of the loop.
            //
            // We can't use an iterator for the loop because `self.nodes` is
            // appended to and the borrow checker would complain. We also can't use
            // `for index in 0..self.nodes.len() { ... }` because the range would
            // be computed with the initial length, and we would miss the appended
            // nodes. Therefore we use a `while` loop.
            let mut index = 0;
            while let Some(node) = self.nodes.get_mut(index) {
                if node.state.get() != NodeState::Pending
                    || !processor.needs_process_obligation(&node.obligation)
                {
                    index += 1;
                    continue;
                }

                // `processor.process_obligation` can modify the predicate within
                // `node.obligation`, and that predicate is the key used for
                // `self.active_cache`. This means that `self.active_cache` can get
                // out of sync with `nodes`. It's not very common, but it does
                // happen, and code in `compress` has to allow for it.

                match processor.process_obligation(&mut node.obligation) {
                    ProcessResult::Unchanged => {
                        // No change in state.
                    }
                    ProcessResult::Changed(children) => {
                        // We are not (yet) stalled.
                        has_changed = true;
                        node.state.set(NodeState::Success);

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
                        has_changed = true;
                        outcome.record_error(Error { error: err, backtrace: self.error_at(index) });
                    }
                }
                index += 1;
            }

            // If unchanged, then we saw no successful obligations, which means
            // there is no point in further iteration. This is based on the
            // assumption that when trait matching returns `Error` or
            // `Unchanged`, those results do not affect environmental inference
            // state. (Note that this will occur if we invoke
            // `process_obligations` with no pending obligations.)
            if !has_changed {
                break;
            }

            self.mark_successes();
            self.process_cycles(processor, &mut outcome);
            self.compress(|obl| outcome.record_completed(obl));
        }

        outcome
    }

    /// Returns a vector of obligations for `p` and all of its
    /// ancestors, putting them into the error state in the process.
    fn error_at(&self, mut index: usize) -> Vec<O> {
        let mut error_stack: Vec<usize> = vec![];
        let mut trace = vec![];

        loop {
            let node = &self.nodes[index];
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
    fn mark_successes(&self) {
        // Convert all `Waiting` nodes to `Success`.
        for node in &self.nodes {
            if node.state.get() == NodeState::Waiting {
                node.state.set(NodeState::Success);
            }
        }

        // Convert `Success` nodes that depend on a pending node back to
        // `Waiting`.
        for node in &self.nodes {
            if node.state.get() == NodeState::Pending {
                // This call site is hot.
                self.inlined_mark_dependents_as_waiting(node);
            }
        }
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
    fn process_cycles<P>(&mut self, processor: &mut P, outcome: &mut P::OUT)
    where
        P: ObligationProcessor<Obligation = O>,
    {
        let mut stack = std::mem::take(&mut self.reused_node_vec);
        for (index, node) in self.nodes.iter().enumerate() {
            // For some benchmarks this state test is extremely hot. It's a win
            // to handle the no-op cases immediately to avoid the cost of the
            // function call.
            if node.state.get() == NodeState::Success {
                self.find_cycles_from_node(&mut stack, processor, index, outcome);
            }
        }

        debug_assert!(stack.is_empty());
        self.reused_node_vec = stack;
    }

    fn find_cycles_from_node<P>(
        &self,
        stack: &mut Vec<usize>,
        processor: &mut P,
        index: usize,
        outcome: &mut P::OUT,
    ) where
        P: ObligationProcessor<Obligation = O>,
    {
        let node = &self.nodes[index];
        if node.state.get() == NodeState::Success {
            match stack.iter().rposition(|&n| n == index) {
                None => {
                    stack.push(index);
                    for &dep_index in node.dependents.iter() {
                        self.find_cycles_from_node(stack, processor, dep_index, outcome);
                    }
                    stack.pop();
                    node.state.set(NodeState::Done);
                }
                Some(rpos) => {
                    // Cycle detected.
                    let result = processor.process_backedge(
                        stack[rpos..].iter().map(|&i| &self.nodes[i].obligation),
                        PhantomData,
                    );
                    if let Err(err) = result {
                        outcome.record_error(Error { error: err, backtrace: self.error_at(index) });
                    }
                }
            }
        }
    }

    /// Compresses the vector, removing all popped nodes. This adjusts the
    /// indices and hence invalidates any outstanding indices. `process_cycles`
    /// must be run beforehand to remove any cycles on `Success` nodes.
    #[inline(never)]
    fn compress(&mut self, mut outcome_cb: impl FnMut(&O)) {
        let orig_nodes_len = self.nodes.len();
        let mut node_rewrites: Vec<_> = std::mem::take(&mut self.reused_node_vec);
        debug_assert!(node_rewrites.is_empty());
        node_rewrites.extend(0..orig_nodes_len);
        let mut dead_nodes = 0;

        // Move removable nodes to the end, preserving the order of the
        // remaining nodes.
        //
        // LOOP INVARIANT:
        //     self.nodes[0..index - dead_nodes] are the first remaining nodes
        //     self.nodes[index - dead_nodes..index] are all dead
        //     self.nodes[index..] are unchanged
        for index in 0..orig_nodes_len {
            let node = &self.nodes[index];
            match node.state.get() {
                NodeState::Pending | NodeState::Waiting => {
                    if dead_nodes > 0 {
                        self.nodes.swap(index, index - dead_nodes);
                        node_rewrites[index] -= dead_nodes;
                    }
                }
                NodeState::Done => {
                    // The removal lookup might fail because the contents of
                    // `self.active_cache` are not guaranteed to match those of
                    // `self.nodes`. See the comment in `process_obligation`
                    // for more details.
                    let cache_key = node.obligation.as_cache_key();
                    self.active_cache.remove(&cache_key);
                    self.done_cache.insert(cache_key);

                    // Extract the success stories.
                    outcome_cb(&node.obligation);
                    node_rewrites[index] = orig_nodes_len;
                    dead_nodes += 1;
                }
                NodeState::Error => {
                    // We *intentionally* remove the node from the cache at this point. Otherwise
                    // tests must come up with a different type on every type error they
                    // check against.
                    self.active_cache.remove(&node.obligation.as_cache_key());
                    self.insert_into_error_cache(index);
                    node_rewrites[index] = orig_nodes_len;
                    dead_nodes += 1;
                }
                NodeState::Success => unreachable!(),
            }
        }

        if dead_nodes > 0 {
            // Remove the dead nodes and rewrite indices.
            self.nodes.truncate(orig_nodes_len - dead_nodes);
            self.apply_rewrites(&node_rewrites);
        }

        node_rewrites.truncate(0);
        self.reused_node_vec = node_rewrites;
    }

    #[inline(never)]
    fn apply_rewrites(&mut self, node_rewrites: &[usize]) {
        let orig_nodes_len = node_rewrites.len();

        for node in &mut self.nodes {
            let mut i = 0;
            while let Some(dependent) = node.dependents.get_mut(i) {
                let new_index = node_rewrites[*dependent];
                if new_index >= orig_nodes_len {
                    node.dependents.swap_remove(i);
                    if i == 0 && node.has_parent {
                        // We just removed the parent.
                        node.has_parent = false;
                    }
                } else {
                    *dependent = new_index;
                    i += 1;
                }
            }
        }

        // This updating of `self.active_cache` is necessary because the
        // removal of nodes within `compress` can fail. See above.
        self.active_cache.retain(|_predicate, index| {
            let new_index = node_rewrites[*index];
            if new_index >= orig_nodes_len {
                false
            } else {
                *index = new_index;
                true
            }
        });
    }
}
