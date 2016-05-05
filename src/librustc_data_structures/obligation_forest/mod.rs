// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `ObligationForest` is a utility data structure used in trait
//! matching to track the set of outstanding obligations (those not
//! yet resolved to success or error). It also tracks the "backtrace"
//! of each pending obligation (why we are trying to figure this out
//! in the first place). See README.md for a general overview of how
//! to use this class.

use fnv::{FnvHashMap, FnvHashSet};

use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash;

mod node_index;
use self::node_index::NodeIndex;

#[cfg(test)]
mod test;

pub trait ForestObligation : Clone {
    type Predicate : Clone + hash::Hash + Eq + ::std::fmt::Debug;

    fn as_predicate(&self) -> &Self::Predicate;
}

pub trait ObligationProcessor {
    type Obligation : ForestObligation;
    type Error : Debug;

    fn process_obligation(&mut self,
                          obligation: &mut Self::Obligation)
                          -> Result<Option<Vec<Self::Obligation>>, Self::Error>;

    fn process_backedge(&mut self, cycle: &[Self::Obligation]);
}

struct SnapshotData {
    node_len: usize,
    cache_list_len: usize,
}

pub struct ObligationForest<O: ForestObligation> {
    /// The list of obligations. In between calls to
    /// `process_obligations`, this list only contains nodes in the
    /// `Pending` or `Success` state (with a non-zero number of
    /// incomplete children). During processing, some of those nodes
    /// may be changed to the error state, or we may find that they
    /// are completed (That is, `num_incomplete_children` drops to 0).
    /// At the end of processing, those nodes will be removed by a
    /// call to `compress`.
    ///
    /// At all times we maintain the invariant that every node appears
    /// at a higher index than its parent. This is needed by the
    /// backtrace iterator (which uses `split_at`).
    nodes: Vec<Node<O>>,
    done_cache: FnvHashSet<O::Predicate>,
    waiting_cache: FnvHashMap<O::Predicate, NodeIndex>,
    cache_list: Vec<O::Predicate>,
    snapshots: Vec<SnapshotData>,
    scratch: Option<Vec<usize>>,
}

pub struct Snapshot {
    len: usize,
}

#[derive(Debug)]
struct Node<O> {
    obligation: O,
    state: NodeState,

    // these both go *in the same direction*.
    parent: Option<NodeIndex>,
    dependants: Vec<NodeIndex>,
}

/// The state of one node in some tree within the forest. This
/// represents the current state of processing for the obligation (of
/// type `O`) associated with this node.
#[derive(Debug, PartialEq, Eq)]
enum NodeState {
    /// Obligation not yet resolved to success or error.
    Pending,

    /// Used before garbage collection
    Success,

    /// Obligation resolved to success; `num_incomplete_children`
    /// indicates the number of children still in an "incomplete"
    /// state. Incomplete means that either the child is still
    /// pending, or it has children which are incomplete. (Basically,
    /// there is pending work somewhere in the subtree of the child.)
    ///
    /// Once all children have completed, success nodes are removed
    /// from the vector by the compression step.
    Waiting,

    /// This obligation, along with its subobligations, are complete,
    /// and will be removed in the next collection.
    Done,

    /// This obligation was resolved to an error. Error nodes are
    /// removed from the vector by the compression step.
    Error,
}

#[derive(Debug)]
pub struct Outcome<O, E> {
    /// Obligations that were completely evaluated, including all
    /// (transitive) subobligations.
    pub completed: Vec<O>,

    /// Backtrace of obligations that were found to be in error.
    pub errors: Vec<Error<O, E>>,

    /// If true, then we saw no successful obligations, which means
    /// there is no point in further iteration. This is based on the
    /// assumption that when trait matching returns `Err` or
    /// `Ok(None)`, those results do not affect environmental
    /// inference state. (Note that if we invoke `process_obligations`
    /// with no pending obligations, stalled will be true.)
    pub stalled: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Error<O, E> {
    pub error: E,
    pub backtrace: Vec<O>,
}

impl<O: Debug + ForestObligation> ObligationForest<O> {
    pub fn new() -> ObligationForest<O> {
        ObligationForest {
            nodes: vec![],
            snapshots: vec![],
            done_cache: FnvHashSet(),
            waiting_cache: FnvHashMap(),
            cache_list: vec![],
            scratch: Some(vec![]),
        }
    }

    /// Return the total number of nodes in the forest that have not
    /// yet been fully resolved.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn start_snapshot(&mut self) -> Snapshot {
        self.snapshots.push(SnapshotData {
            node_len: self.nodes.len(),
            cache_list_len: self.cache_list.len()
        });
        Snapshot { len: self.snapshots.len() }
    }

    pub fn commit_snapshot(&mut self, snapshot: Snapshot) {
        assert_eq!(snapshot.len, self.snapshots.len());
        let info = self.snapshots.pop().unwrap();
        assert!(self.nodes.len() >= info.node_len);
        assert!(self.cache_list.len() >= info.cache_list_len);
    }

    pub fn rollback_snapshot(&mut self, snapshot: Snapshot) {
        // Check that we are obeying stack discipline.
        assert_eq!(snapshot.len, self.snapshots.len());
        let info = self.snapshots.pop().unwrap();

        for entry in &self.cache_list[info.cache_list_len..] {
            self.done_cache.remove(entry);
            self.waiting_cache.remove(entry);
        }

        self.nodes.truncate(info.node_len);
        self.cache_list.truncate(info.cache_list_len);
    }

    pub fn in_snapshot(&self) -> bool {
        !self.snapshots.is_empty()
    }

    /// Registers an obligation
    ///
    /// This CAN be done in a snapshot
    pub fn register_obligation(&mut self, obligation: O) {
        self.register_obligation_at(obligation, None)
    }

    fn register_obligation_at(&mut self, obligation: O, parent: Option<NodeIndex>) {
        if self.done_cache.contains(obligation.as_predicate()) { return }

        match self.waiting_cache.entry(obligation.as_predicate().clone()) {
            Entry::Occupied(o) => {
                debug!("register_obligation_at({:?}, {:?}) - duplicate of {:?}!",
                       obligation, parent, o.get());
                if let Some(parent) = parent {
                    self.nodes[o.get().get()].dependants.push(parent);
                }
            }
            Entry::Vacant(v) => {
                debug!("register_obligation_at({:?}, {:?}) - ok",
                       obligation, parent);
                v.insert(NodeIndex::new(self.nodes.len()));
                self.cache_list.push(obligation.as_predicate().clone());
                self.nodes.push(Node::new(parent, obligation));
            }
        };
    }

    /// Convert all remaining obligations to the given error.
    ///
    /// This cannot be done during a snapshot.
    pub fn to_errors<E: Clone>(&mut self, error: E) -> Vec<Error<O, E>> {
        assert!(!self.in_snapshot());
        let mut errors = vec![];
        for index in 0..self.nodes.len() {
            debug_assert!(!self.nodes[index].is_popped());
            if let NodeState::Pending = self.nodes[index].state {
                let backtrace = self.error_at(index);
                errors.push(Error {
                    error: error.clone(),
                    backtrace: backtrace,
                });
            }
        }
        let successful_obligations = self.compress();
        assert!(successful_obligations.is_empty());
        errors
    }

    /// Returns the set of obligations that are in a pending state.
    pub fn pending_obligations(&self) -> Vec<O>
        where O: Clone
    {
        self.nodes
            .iter()
            .filter(|n| n.state == NodeState::Pending)
            .map(|n| n.obligation.clone())
            .collect()
    }

    /// Perform a pass through the obligation list. This must
    /// be called in a loop until `outcome.stalled` is false.
    ///
    /// This CANNOT be unrolled (presently, at least).
    pub fn process_obligations<P>(&mut self, processor: &mut P) -> Outcome<O, P::Error>
        where P: ObligationProcessor<Obligation=O>
    {
        debug!("process_obligations(len={})", self.nodes.len());
        assert!(!self.in_snapshot()); // cannot unroll this action

        let mut errors = vec![];
        let mut stalled = true;

        for index in 0..self.nodes.len() {
            debug_assert!(!self.nodes[index].is_popped());

            debug!("process_obligations: node {} == {:?}",
                   index,
                   self.nodes[index]);

            let result = match self.nodes[index] {
                Node { state: NodeState::Pending, ref mut obligation, .. } => {
                    processor.process_obligation(obligation)
                }
                _ => continue
            };

            debug!("process_obligations: node {} got result {:?}",
                   index,
                   result);

            match result {
                Ok(None) => {
                    // no change in state
                }
                Ok(Some(children)) => {
                    // if we saw a Some(_) result, we are not (yet) stalled
                    stalled = false;
                    for child in children {
                        self.register_obligation_at(child,
                                                    Some(NodeIndex::new(index)));
                    }

                    self.nodes[index].state = NodeState::Success;
                }
                Err(err) => {
                    let backtrace = self.error_at(index);
                    errors.push(Error {
                        error: err,
                        backtrace: backtrace,
                    });
                }
            }
        }

        self.mark_as_waiting();
        self.process_cycles(processor);

        // Now we have to compress the result
        let completed_obligations = self.compress();

        debug!("process_obligations: complete");

        Outcome {
            completed: completed_obligations,
            errors: errors,
            stalled: stalled,
        }
    }

    pub fn process_cycles<P>(&mut self, _processor: &mut P)
        where P: ObligationProcessor<Obligation=O>
    {
        // TODO: implement
        for node in &mut self.nodes {
            if node.state == NodeState::Success {
                node.state = NodeState::Done;
            }
        }
    }

    /// Returns a vector of obligations for `p` and all of its
    /// ancestors, putting them into the error state in the process.
    /// The fact that the root is now marked as an error is used by
    /// `inherit_error` above to propagate the error state to the
    /// remainder of the tree.
    fn error_at(&mut self, p: usize) -> Vec<O> {
        let mut error_stack = self.scratch.take().unwrap();
        let mut trace = vec![];

        let mut n = p;
        loop {
            self.nodes[n].state = NodeState::Error;
            trace.push(self.nodes[n].obligation.clone());
            error_stack.extend(self.nodes[n].dependants.iter().map(|x| x.get()));

            // loop to the parent
            match self.nodes[n].parent {
                Some(q) => n = q.get(),
                None => break
            }
        }

        loop {
            // non-standard `while let` to bypass #6393
            let i = match error_stack.pop() {
                Some(i) => i,
                None => break
            };

            match self.nodes[i].state {
                NodeState::Error => continue,
                ref mut s => *s = NodeState::Error
            }

            let node = &self.nodes[i];
            error_stack.extend(
                node.dependants.iter().cloned().chain(node.parent).map(|x| x.get())
            );
        }

        self.scratch = Some(error_stack);
        trace
    }

    fn mark_as_waiting(&mut self) {
        for node in &mut self.nodes {
            if node.state == NodeState::Waiting {
                node.state = NodeState::Success;
            }
        }

        let mut undone_stack = self.scratch.take().unwrap();
        undone_stack.extend(
            self.nodes.iter().enumerate()
                .filter(|&(_i, n)| n.state == NodeState::Pending)
                .map(|(i, _n)| i));

        loop {
            // non-standard `while let` to bypass #6393
            let i = match undone_stack.pop() {
                Some(i) => i,
                None => break
            };

            match self.nodes[i].state {
                NodeState::Pending | NodeState::Done => {},
                NodeState::Waiting | NodeState::Error => continue,
                ref mut s @ NodeState::Success => {
                    *s = NodeState::Waiting;
                }
            }

            let node = &self.nodes[i];
            undone_stack.extend(
                node.dependants.iter().cloned().chain(node.parent).map(|x| x.get())
            );
        }

        self.scratch = Some(undone_stack);
    }

    /// Compresses the vector, removing all popped nodes. This adjusts
    /// the indices and hence invalidates any outstanding
    /// indices. Cannot be used during a transaction.
    ///
    /// Beforehand, all nodes must be marked as `Done` and no cycles
    /// on these nodes may be present. This is done by e.g. `process_cycles`.
    #[inline(never)]
    fn compress(&mut self) -> Vec<O> {
        assert!(!self.in_snapshot()); // didn't write code to unroll this action

        let nodes_len = self.nodes.len();
        let mut node_rewrites: Vec<_> = self.scratch.take().unwrap();
        node_rewrites.extend(0..nodes_len);
        let mut dead_nodes = 0;

        // Now move all popped nodes to the end. Try to keep the order.
        //
        // LOOP INVARIANT:
        //     self.nodes[0..i - dead_nodes] are the first remaining nodes
        //     self.nodes[i - dead_nodes..i] are all dead
        //     self.nodes[i..] are unchanged
        for i in 0..self.nodes.len() {
            if let NodeState::Done = self.nodes[i].state {
                self.done_cache.insert(self.nodes[i].obligation.as_predicate().clone());
            }

            if self.nodes[i].is_popped() {
                self.waiting_cache.remove(self.nodes[i].obligation.as_predicate());
                node_rewrites[i] = nodes_len;
                dead_nodes += 1;
            } else {
                if dead_nodes > 0 {
                    self.nodes.swap(i, i - dead_nodes);
                    node_rewrites[i] -= dead_nodes;
                }
            }
        }

        // No compression needed.
        if dead_nodes == 0 {
            node_rewrites.truncate(0);
            self.scratch = Some(node_rewrites);
            return vec![];
        }

        // Pop off all the nodes we killed and extract the success
        // stories.
        let successful = (0..dead_nodes)
                             .map(|_| self.nodes.pop().unwrap())
                             .flat_map(|node| {
                                 match node.state {
                                     NodeState::Error => None,
                                     NodeState::Done => Some(node.obligation),
                                     _ => unreachable!()
                                 }
                             })
            .collect();
        self.apply_rewrites(&node_rewrites);

        node_rewrites.truncate(0);
        self.scratch = Some(node_rewrites);

        successful
    }

    fn apply_rewrites(&mut self, node_rewrites: &[usize]) {
        let nodes_len = node_rewrites.len();

        for node in &mut self.nodes {
            if let Some(index) = node.parent {
                let new_index = node_rewrites[index.get()];
                if new_index >= nodes_len {
                    // parent dead due to error
                    node.parent = None;
                } else {
                    node.parent = Some(NodeIndex::new(new_index));
                }
            }

            let mut i = 0;
            while i < node.dependants.len() {
                let new_index = node_rewrites[node.dependants[i].get()];
                if new_index >= nodes_len {
                    node.dependants.swap_remove(i);
                } else {
                    node.dependants[i] = NodeIndex::new(new_index);
                    i += 1;
                }
            }
        }

        let mut kill_list = vec![];
        for (predicate, index) in self.waiting_cache.iter_mut() {
            let new_index = node_rewrites[index.get()];
            if new_index >= nodes_len {
                kill_list.push(predicate.clone());
            } else {
                *index = NodeIndex::new(new_index);
            }
        }

        for predicate in kill_list { self.waiting_cache.remove(&predicate); }
    }
}

impl<O> Node<O> {
    fn new(parent: Option<NodeIndex>, obligation: O) -> Node<O> {
        Node {
            obligation: obligation,
            parent: parent,
            state: NodeState::Pending,
            dependants: vec![],
        }
    }

    fn is_popped(&self) -> bool {
        match self.state {
            NodeState::Pending | NodeState::Success | NodeState::Waiting => false,
            NodeState::Error | NodeState::Done => true,
        }
    }
}
