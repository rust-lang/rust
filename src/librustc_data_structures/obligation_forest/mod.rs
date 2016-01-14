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

use std::cmp::Ordering;
use std::fmt::Debug;
use std::mem;

mod node_index;

#[cfg(test)]
mod test;

#[derive(Debug)]
pub struct ObligationForest<O> {
    /// The list of obligations. In between calls to
    /// `process_obligations`, this list only contains nodes in the
    /// `Pending` or `Success` state (with a non-zero number of
    /// incomplete children). During processing, some of those nodes
    /// may be changed to the error state, or we may find that they
    /// are completed (That is, `num_incomplete_children` drops to 0).
    /// At the end of processing, those nodes will be removed (or
    /// marked as removed if used in earlier snapshots) by a call to
    /// `compress`.
    ///
    /// At all times we maintain the invariant that every node appears
    /// at a higher index than its parent. This is needed by the
    /// backtrace iterator (which uses `split_at`).
    nodes: Vec<Node<O>>,
    snapshots: Vec<Snapshot>
}

// We could implement Copy here, but that tastes weird.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Snapshot(usize);

pub use self::node_index::NodeIndex;

/// Snapshotted state of a node. The snapshot is the one in which the node state *applies*, not the
/// one upon which the node state *was applied*.
#[derive(Debug)]
struct NodeStateSnapshot<O> {
    snapshot: Snapshot,
    state: NodeState<O>,
}
impl<O> NodeStateSnapshot<O> {
    fn new(snapshot: Snapshot, state: NodeState<O>) -> NodeStateSnapshot<O> {
        NodeStateSnapshot {
            snapshot: snapshot,
            state: state,
        }
    }
}

/// Stack of node state snapshots. Has implicit stack structure from the total ordering of the
/// Snapshot structure and its closure under Snapshot::next(): every stack has an implicit value
/// for every possible snapshot, with the most recent state at or below any given snapshot being
/// the applicable state.
///
/// The stack is split into a Base and a Stack variant to take advantage of the per-node
/// snapshotting performed by ObligationForest and to thus avoid allocating space for snapshots
/// when it isn't necessary.
#[derive(Debug)]
enum NodeStateSnapshots<O> {
    Base(NodeStateSnapshot<O>),
    Stack(Vec<NodeStateSnapshot<O>>),
}

#[derive(Debug)]
struct Node<O> {
    snapshots: NodeStateSnapshots<O>,
    parent: Option<NodeIndex>,
    root: NodeIndex, // points to the root, which may be the current node
}

// FIXME make the `obligation` member a `Cow<O>` instead of an `O` to avoid cloning so much (or,
// maybe don't? Maybe the extra byte for the discriminant isn't worth the usually inexpensive
// clone?). Maybe move the Obligation into the Node instead?

/// The state of one node in some tree within the forest. This
/// represents the current state of processing for the obligation (of
/// type `O`) associated with this node.
#[derive(Debug)]
enum NodeState<O> {
    /// Obligation not yet resolved to success or error.
    Pending { obligation: O },

    /// Obligation resolved to success; `num_incomplete_children`
    /// indicates the number of children still in an "incomplete"
    /// state. Incomplete means that either the child is still
    /// pending, or it has children which are incomplete. (Basically,
    /// there is pending work somewhere in the subtree of the child.)
    ///
    /// Once all children have completed, success nodes are removed
    /// from the vector by the compression step if they have no
    /// underlying snapshots that are still alive. Else, they're set
    /// to 'Popped'.
    Success { obligation: O, num_incomplete_children: usize },

    /// This obligation was resolved to an error. Error nodes are
    /// removed from the vector by the compression step if they have
    /// no underlying snapshots that are still alive. Else, they're
    /// set to 'Popped'.
    Error,

    /// Obligation is dead in the latest snapshot, but still being
    /// used by earlier snapshots. Note that this state may occur
    /// even if there are no earlier snapshots if a node became
    /// Popped then was committed to a lower snapshot.
    Popped,
}

#[derive(Debug)]
pub struct Outcome<O,E> {
    /// Obligations that were completely evaluated, including all
    /// (transitive) subobligations.
    pub completed: Vec<O>,

    /// Backtrace of obligations that were found to be in error.
    pub errors: Vec<Error<O,E>>,

    /// If true, then we saw no successful obligations, which means
    /// there is no point in further iteration. This is based on the
    /// assumption that when trait matching returns `Err` or
    /// `Ok(None)`, those results do not affect environmental
    /// inference state. (Note that if we invoke `process_obligations`
    /// with no pending obligations, stalled will be true.)
    pub stalled: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Error<O,E> {
    pub error: E,
    pub backtrace: Vec<O>,
}

impl<O: Debug + Clone> ObligationForest<O> {
    pub fn new() -> ObligationForest<O> {
        ObligationForest {
            nodes: vec![],
            snapshots: vec![Snapshot::new_base()]
        }
    }

    /// Return the total number of nodes in the forest that have not
    /// yet been fully resolved.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    fn current_snapshot(&self) -> Snapshot {
        self.snapshots.last().unwrap().clone()
    }

    pub fn start_snapshot(&mut self) -> Snapshot {
        let next_snapshot = self.snapshots.last().unwrap().next();
        self.snapshots.push(next_snapshot.clone());
        next_snapshot
    }

    pub fn commit_snapshot(&mut self, snapshot: Snapshot) {
        assert_eq!(*self.snapshots.last().unwrap(), snapshot);
        let prev_snapshot = snapshot.prev();
        for node in &mut self.nodes {
            node.snapshots.commit(snapshot.clone(), prev_snapshot.clone());
        }
        self.snapshots.pop();
    }

    pub fn rollback_snapshot(&mut self, snapshot: Snapshot) {
        assert_eq!(*self.snapshots.last().unwrap(), snapshot);
        for node in &mut self.nodes {
            node.snapshots.pop(snapshot.clone());
        }
        // Compress before popping the snapshot to ensure that the
        // snapshot seen while popping is the one that corresponds
        // to snapshotted Popped states.
        self.compress();
        self.snapshots.pop();
        assert!(!self.snapshots.is_empty(), "rolled back into non-existence");
    }

    pub fn in_snapshot(&self) -> bool {
        self.snapshots.len() != 1
    }

    /// Adds a new tree to the forest.
    ///
    /// This CAN be done during a snapshot.
    pub fn push_root(&mut self, obligation: O) {
        let index = NodeIndex::new(self.nodes.len());
        let current_snapshot = self.current_snapshot();
        self.nodes.push(Node::new(current_snapshot, index, None, obligation));
    }

    /// Convert all remaining obligations to the given error.
    pub fn to_errors<E:Clone>(&mut self, error: E) -> Vec<Error<O,E>> {
        let mut errors = vec![];
        for index in 0..self.nodes.len() {
            debug_assert!(!self.nodes[index].is_popped_at(self.current_snapshot()));
            self.inherit_error(index);
            if let &NodeState::Pending { .. } = self.nodes[index].snapshots.top() {
                let backtrace = self.backtrace(index);
                errors.push(Error { error: error.clone(), backtrace: backtrace });
            }
        }
        let successful_obligations = self.compress();
        assert!(successful_obligations.is_empty());
        errors
    }

    /// Returns the set of obligations that are in a pending state.
    pub fn pending_obligations(&self) -> Vec<O> where O: Clone {
        self.nodes.iter()
                  .filter_map(|n| match n.snapshots.top() {
                      &NodeState::Pending { ref obligation } => Some(obligation),
                      _ => None,
                  })
                  .cloned()
                  .collect()
    }

    /// Process the obligations.
    pub fn process_obligations<E,F>(&mut self, mut action: F) -> Outcome<O,E>
        where E: Debug, F: FnMut(&mut O, Backtrace<O>) -> Result<Option<Vec<O>>, E>
    {
        debug!("process_obligations(len={})", self.nodes.len());

        let mut errors = vec![];
        let mut stalled = true;

        // We maintain the invariant that the list is in pre-order, so
        // parents occur before their children. Also, whenever an
        // error occurs, we propagate it from the child all the way to
        // the root of the tree. Together, these two facts mean that
        // when we visit a node, we can check if its root is in error,
        // and we will find out if any prior node within this forest
        // encountered an error.

        for index in 0..self.nodes.len() {
            if self.nodes[index].is_popped_at(self.current_snapshot()) {
                // FIXME have compression move nodes that are popped at the current snapshot to the
                // end of the nodes array, s.t. if we started keeping track of the total number of
                // nodes alive in this snapshot, we could always skip the popped ones instead of
                // explicitly checking.
                continue;
            }
            self.inherit_error(index);

            debug!("process_obligations: node {} == {:?}",
                   index, self.nodes[index].snapshots.top());

            let result = {
                let parent = self.nodes[index].parent;
                let (prefix, suffix) = self.nodes.split_at_mut(index);
                let backtrace = Backtrace::new(prefix, parent);
                match suffix[0].snapshots.top_mut() {
                    &mut NodeState::Popped |
                    &mut NodeState::Error |
                    &mut NodeState::Success { .. } =>
                        continue,
                    &mut NodeState::Pending { ref mut obligation } =>
                        action(obligation, backtrace),
                }
            };

            debug!("process_obligations: node {} got result {:?}", index, result);

            match result {
                Ok(None) => {
                    // no change in state
                }
                Ok(Some(children)) => {
                    // if we saw a Some(_) result, we are not (yet) stalled
                    stalled = false;
                    self.success(index, children);
                }
                Err(err) => {
                    let backtrace = self.backtrace(index);
                    errors.push(Error { error: err, backtrace: backtrace });
                }
            }
        }

        // Now we have to compress the result
        let successful_obligations = self.compress();

        debug!("process_obligations: complete");

        Outcome {
            completed: successful_obligations,
            errors: errors,
            stalled: stalled,
        }
    }

    /// Indicates that node `index` has been processed successfully,
    /// yielding `children` as the derivative work. If children is an
    /// empty vector, this will update the ref count on the parent of
    /// `index` to indicate that a child has completed
    /// successfully. Otherwise, adds new nodes to represent the child
    /// work.
    fn success(&mut self, index: usize, children: Vec<O>) {
        debug!("success(index={}, children={:?})", index, children);

        let num_incomplete_children = children.len();
        let current_snapshot = self.current_snapshot();

        if num_incomplete_children == 0 {
            // if there is no work left to be done, decrement parent's ref count
            self.update_parent(index);
        } else {
            // create child work
            let root_index = self.nodes[index].root;
            let node_index = NodeIndex::new(index);
            self.nodes.extend(
                children.into_iter()
                        .map(|o| Node::new(current_snapshot.clone(),
                                           root_index,
                                           Some(node_index),
                                           o)));
        }

        // change state from `Pending` to `Success`
        self.nodes[index].snapshots.update_with(current_snapshot.clone(), |_, state| match state {
            &NodeState::Pending { ref obligation } =>
                Some(NodeState::Success { obligation: obligation.clone(),
                                          num_incomplete_children: num_incomplete_children }),
            &NodeState::Success { .. } |
            &NodeState::Error |
            &NodeState::Popped =>
                unreachable!()
        });
    }

    /// Decrements the ref count on the parent of `child`; if the
    /// parent's ref count then reaches zero, proceeds recursively.
    fn update_parent(&mut self, child: usize) {
        debug!("update_parent(child={})", child);
        let current_snapshot = self.current_snapshot();
        if let Some(parent) = self.nodes[child].parent {
            let parent = parent.get();
            let mut skip_update_parent = false;
            self.nodes[parent].snapshots.update_with(
                current_snapshot,
                |_, state| match state {
                    &NodeState::Success { ref num_incomplete_children, ref obligation } => {
                        if *num_incomplete_children > 1 {
                            skip_update_parent = true;
                        }
                        Some(NodeState::Success {
                            num_incomplete_children: num_incomplete_children - 1,
                            obligation: obligation.clone()
                        })
                    }
                    _ => unreachable!(),
                });
            if !skip_update_parent {
                self.update_parent(parent);
            }
        }
    }

    /// If the root of `child` is in an error state, places `child`
    /// into an error state. This is used during processing so that we
    /// skip the remaining obligations from a tree once some other
    /// node in the tree is found to be in error.
    fn inherit_error(&mut self, child: usize) {
        let current_snapshot = self.current_snapshot();
        let root = self.nodes[child].root.get();
        if let NodeState::Error = *self.nodes[root].snapshots.top() {
            self.nodes[child].snapshots.update(current_snapshot, NodeState::Error);
        }
    }

    /// Returns a vector of obligations for `p` and all of its
    /// ancestors, putting them into the error state in the process.
    /// The fact that the root is now marked as an error is used by
    /// `inherit_error` above to propagate the error state to the
    /// remainder of the tree.
    fn backtrace(&mut self, mut p: usize) -> Vec<O> {
        let mut trace = vec![];
        let current_snapshot = self.current_snapshot();
        loop {
            self.nodes[p].snapshots.update_with(current_snapshot.clone(), |_, state| {
                match state {
                    &NodeState::Pending { ref obligation } |
                    &NodeState::Success { ref obligation, .. } => {
                        trace.push(obligation.clone());
                    }
                    &NodeState::Error => {
                        // we should not encounter an error, because if
                        // there was an error in the ancestors, it should
                        // have been propagated down and we should never
                        // have tried to process this obligation
                        panic!("encountered error in node {:?} when collecting stack trace", p);
                    }
                    &NodeState::Popped => {
                        // We should not encounter a popped node, because if there was a popped
                        // node in the ancestors, it would mean that it was done being used, but
                        // we're backtracing from a still-in-use node so all parent nodes must be
                        // aware of this and be in use.
                        panic!("encountered dead node {:?} when collecting stack trace", p);
                    }
                }
                Some(NodeState::Error)
            });

            // loop to the parent
            match self.nodes[p].parent {
                Some(q) => { p = q.get(); }
                None => { return trace; }
            }
        }
    }

    /// Compresses the vector, removing all popped nodes. This adjusts
    /// the indices and hence invalidates any outstanding
    /// indices.
    fn compress(&mut self) -> Vec<O> {
        // FIXME have compression move nodes that are popped at the current snapshot to the
        // end of the nodes array but *before the dead nodes*, s.t. if we started keeping track of
        // the total number of nodes alive in this snapshot, we could always skip the popped ones.
        // Or maybe the boatload of swapping is more expensive than just sifting through? I'unno.

        let mut rewrites: Vec<_> = (0..self.nodes.len()).collect();
        let current_snapshot = self.current_snapshot();

        // Finish propagating error state. Note that in this case we
        // only have to check immediate parents, rather than all
        // ancestors, because all errors have already occurred that
        // are going to occur.
        let nodes_len = self.nodes.len();
        for i in 0..nodes_len {
            if !self.nodes[i].is_popped_at(current_snapshot.clone()) {
                self.inherit_error(i);
            }
        }

        // Now go through and move all nodes that are either
        // successful or which have an error over to the end of the
        // list, preserving the relative order of the survivors
        // (which is important for the `inherit_error` logic).
        //
        // Note that even in the presence of snapshots this maintains
        // the pre-ordering of the nodes.
        let mut dead = 0;
        for i in 0..nodes_len {
            if self.nodes[i].is_dead(current_snapshot.clone()) {
                dead += 1;
            } else if dead > 0 {
                self.nodes.swap(i, i - dead);
                rewrites[i] -= dead;
            }
        }

        // Pop off all the nodes we killed and extract the success
        // stories.
        let successful_dead: Vec<_> =
            (0 .. dead).map(|_| self.nodes.pop().unwrap())
                       .flat_map(|node| match node.snapshots.top() {
                           &NodeState::Pending { .. } => unreachable!(),
                           &NodeState::Popped |
                           &NodeState::Error => None,
                           &NodeState::Success { ref obligation, num_incomplete_children } => {
                               assert_eq!(num_incomplete_children, 0);
                               // FIXME introduce a way of just moving the obligation out of the
                               // top snapshot without triggering a panic.
                               Some(obligation.clone())
                           }
                       })
                       .collect();
        // Go through the still-alive successful and error nodes and extract the success stories;
        // they are then marked in the current snapshot as 'Popped' so that they won't be
        // reported again.
        let mut successful: Vec<_> =
            self.nodes.iter_mut()
                      .flat_map(|node| {
                          let mut is_popped = false;
                          let result = match node.snapshots.top() {
                              &NodeState::Pending { .. } => None,
                              &NodeState::Error => { is_popped = true; None },
                              &NodeState::Success { ref obligation, num_incomplete_children } => {
                                  if num_incomplete_children == 0 {
                                      is_popped = true;
                                      Some(obligation.clone())
                                  } else {
                                      None
                                  }
                              },
                              &NodeState::Popped => None,
                          };
                          if is_popped {
                              node.snapshots.update(current_snapshot.clone(), NodeState::Popped);
                          }
                          result
                      })
                      .collect();
        successful.extend(successful_dead);

        // Adjust the parent indices, since we compressed things.
        for node in &mut self.nodes {
            if let Some(ref mut index) = node.parent {
                let new_index = rewrites[index.get()];
                debug_assert!(new_index < (nodes_len - dead));
                *index = NodeIndex::new(new_index);
            }
            node.root = NodeIndex::new(rewrites[node.root.get()]);
        }

        successful
    }
}

impl Snapshot {
    fn new_base() -> Snapshot { Snapshot(0) }
    fn next(&self) -> Snapshot { Snapshot(self.0 + 1) }
    fn prev(&self) -> Snapshot { assert!(self.0 > 0); Snapshot(self.0 - 1) }
}

impl<O> NodeStateSnapshots<O> {
    fn new(snapshot: Snapshot, state: NodeState<O>) -> NodeStateSnapshots<O> {
        NodeStateSnapshots::Base(NodeStateSnapshot::new(snapshot, state))
    }

    fn top_snapshot(&self) -> Snapshot {
        match self {
            &NodeStateSnapshots::Base(ref base) => base.snapshot.clone(),
            &NodeStateSnapshots::Stack(ref stack) => stack.last().unwrap().snapshot.clone()
        }
    }
    fn top(&self) -> &NodeState<O> {
        match self {
            &NodeStateSnapshots::Base(ref base) => &base.state,
            &NodeStateSnapshots::Stack(ref stack) => &stack.last().unwrap().state
        }
    }
    fn top_mut(&mut self) -> &mut NodeState<O> {
        match self {
            &mut NodeStateSnapshots::Base(ref mut base) => &mut base.state,
            &mut NodeStateSnapshots::Stack(ref mut stack) => &mut stack.last_mut().unwrap().state
        }
    }
    fn len(&self) -> usize {
        match self {
            &NodeStateSnapshots::Base(_) => 1,
            &NodeStateSnapshots::Stack(ref stack) => stack.len()
        }
    }

    /// Pops the given snapshot off of the top of the stack. If the snapshot we have at the top is
    /// from earlier than the snapshot passed to us, we ignore the pop. If the snapshot we have is
    /// greater than the snapshot passed to us, we were somehow skipped in the rest of this code
    /// and we panic.
    fn pop(&mut self, snapshot: Snapshot) -> Option<NodeState<O>> {
        match self {
            &mut NodeStateSnapshots::Stack(ref mut stack) => {
                match stack.last().unwrap().snapshot.cmp(&snapshot) {
                    Ordering::Equal => {
                        let NodeStateSnapshot {snapshot: node_snapshot, state: node_state} =
                            stack.pop().unwrap();
                        if stack.is_empty() {
                            stack.push(NodeStateSnapshot::new(node_snapshot, NodeState::Popped));
                        }
                        Some(node_state)
                    },
                    Ordering::Less => None,
                    Ordering::Greater => panic!("failure to maintain stack discipline")
                }
            },
            &mut NodeStateSnapshots::Base(ref mut base) =>
                match base.snapshot.cmp(&snapshot) {
                    Ordering::Equal => Some(mem::replace(&mut base.state, NodeState::Popped)),
                    Ordering::Less => None,
                    Ordering::Greater => panic!("failure to maintain stack discipline")
                }
        }
    }

    /// Commits from the `from` snapshot down into the `into` snapshot. If the snapshot we have at
    /// the top is at or earlier than the `into` snapshot passed to us, we don't do anything. If
    /// the snapshot we have at the top is later than the `from` snapshot passed to us, we were
    /// somehow skipped in the rest of this code and we panic. Else, we overwrite any snapshot at
    /// the into_snapshot with the from_snapshot state (or make such a state if it doesn't exist by
    /// rewriting only the snapshot at the top of the stack).
    fn commit(&mut self, from_snapshot: Snapshot, into_snapshot: Snapshot) {
        assert!(from_snapshot.prev() == into_snapshot, "failure to maintain stack discipline");
        if self.top_snapshot() <= into_snapshot {
            return;
        }
        if self.top_snapshot() > from_snapshot {
            panic!("failure to maintain stack discipline");
        }
        match self {
            &mut NodeStateSnapshots::Stack(ref mut stack) => {
                let mut to_commit = stack.pop().unwrap();
                if stack.is_empty() {
                } else {
                    let mut do_push = false;
                    {
                        let potential_overwrite = stack.last_mut().unwrap();
                        if potential_overwrite.snapshot == into_snapshot {
                            potential_overwrite.state = mem::replace(&mut to_commit.state,
                                                                     NodeState::Popped);
                        } else {
                            do_push = true;
                        }
                    }
                    if do_push {
                        stack.push(NodeStateSnapshot::new(
                                into_snapshot,
                                mem::replace(&mut to_commit.state, NodeState::Popped)));
                    }
                }
            },
            &mut NodeStateSnapshots::Base(ref mut base) => {
                base.snapshot = into_snapshot;
            }
        }
    }

    /// Optionally updates the given snapshot at the top of the stack. If the snapshot we have at
    /// the top is from earlier than the snapshot passed to us, we push the new state. If the
    /// snapshot we have is from later than the snapshot passed to us, we were somehow skipped in
    /// the rest of this code and we panic.
    ///
    /// The new_state_fn accepts a clone of the latest snapshot on the stack and that snapshot's
    /// state. It returns Some() if the state at the snapshot passed into `update_with` is to be
    /// updated, or None if it need not be updated (e.g. to avoid unnecessarily duplicating node
    /// states and allocations).
    fn update_with<F>(&mut self, snapshot: Snapshot, new_state_fn: F) -> Option<NodeState<O>>
        where F: FnOnce(Snapshot, &NodeState<O>) -> Option<NodeState<O>>
    {
        let mut old_state = None;
        let mut rebase = None;
        match self {
            &mut NodeStateSnapshots::Stack(ref mut stack) => {
                let snapshot_cmp = stack.last().unwrap().snapshot.cmp(&snapshot);
                let new_state = {
                    let state_snapshot = stack.last_mut().unwrap();
                    new_state_fn(state_snapshot.snapshot.clone(),
                                 &state_snapshot.state)
                };
                if new_state.is_none() {
                    // we have nothing to do here
                    return None;
                }
                match snapshot_cmp {
                    Ordering::Equal => {
                        old_state = Some(
                            mem::replace(
                                stack.last_mut().unwrap(),
                                NodeStateSnapshot::new(snapshot.clone(),
                                                       new_state.unwrap())).state);
                    },
                    Ordering::Less => {
                        stack.push(NodeStateSnapshot::new(snapshot.clone(), new_state.unwrap()));
                    },
                    Ordering::Greater => panic!("failure to maintain stack discipline")
                }
            },
            &mut NodeStateSnapshots::Base(ref mut base) => {
                let new_state = new_state_fn(base.snapshot.clone(), &base.state);
                if new_state.is_none() {
                    // we have nothing to do here
                    return None;
                }
                match base.snapshot.cmp(&snapshot) {
                    Ordering::Equal => {
                        old_state = Some(mem::replace(&mut base.state, new_state.unwrap()));
                    },
                    Ordering::Less => {
                        rebase = Some((mem::replace(base, unsafe { mem::uninitialized() }),
                                       new_state.unwrap()));
                    },
                    Ordering::Greater => panic!("failure to maintain stack discipline")
                }
            },
        }
        if let Some((base, new_state)) = rebase {
            let new_self = NodeStateSnapshots::Stack(
                vec![base, NodeStateSnapshot::new(snapshot.clone(), new_state)]);
            mem::forget(mem::replace(self, new_self));
        }
        old_state
    }
    fn update(&mut self, snapshot: Snapshot, new_state: NodeState<O>) -> Option<NodeState<O>> {
        self.update_with(snapshot, move |_, _| Some(new_state))
    }
}

impl<O> Node<O> {
    fn new(snapshot: Snapshot, root: NodeIndex, parent: Option<NodeIndex>, obligation: O)
        -> Node<O>
    {
        Node {
            parent: parent,
            snapshots: NodeStateSnapshots::new(snapshot,
                                               NodeState::Pending { obligation: obligation }),
            root: root
        }
    }

    /// Whether or not this node is popped in all snapshots and is not being used by prior
    /// snapshots.
    fn is_dead(&self, snapshot: Snapshot) -> bool {
        assert!(self.snapshots.top_snapshot() <= snapshot, "failure to maintain stack discpline");
        if self.snapshots.len() > 1 {
            false
        } else {
            self.is_popped_at(snapshot)
        }
    }

    /// Whether or not we're done using this node at the given snapshot and it is popped or is
    /// about to be popped.
    fn is_popped_at(&self, snapshot: Snapshot) -> bool {
        assert!(self.snapshots.top_snapshot() <= snapshot, "failure to maintain stack discipline");
        match self.snapshots.top() {
            &NodeState::Pending { .. } => false,
            &NodeState::Success { num_incomplete_children, .. } => num_incomplete_children == 0,
            &NodeState::Error => true,
            &NodeState::Popped => true,
        }
    }
}


#[derive(Clone)]
pub struct Backtrace<'b, O: 'b> {
    nodes: &'b [Node<O>],
    pointer: Option<NodeIndex>,
}

impl<'b, O> Backtrace<'b, O> {
    fn new(nodes: &'b [Node<O>], pointer: Option<NodeIndex>) -> Backtrace<'b, O> {
        Backtrace { nodes: nodes, pointer: pointer }
    }
}

impl<'b, O> Iterator for Backtrace<'b, O> {
    type Item = &'b O;

    fn next(&mut self) -> Option<&'b O> {
        debug!("Backtrace: self.pointer = {:?}", self.pointer);
        if let Some(p) = self.pointer {
            self.pointer = self.nodes[p.get()].parent;
            match self.nodes[p.get()].snapshots.top() {
                &NodeState::Pending { ref obligation } |
                &NodeState::Success { ref obligation, .. } => {
                    Some(obligation)
                }
                &NodeState::Error => {
                    panic!("Backtrace encountered an error.");
                }
                &NodeState::Popped => {
                    panic!("Backtrace encountered a popped node.");
                }
            }
        } else {
            None
        }
    }
}
