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
    snapshots: Vec<Snapshot>,
}

// We could implement Copy here, but we only ever expect user code to consume the snapshot exactly
// once.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Snapshot {
    len: usize
}

pub use self::node_index::NodeIndex;

/// We take advantage of the acyclic state transitions of a node (sans rollbacks)...
///
/// Pending -+-> Success --+
///          |             |
///          +-> Error <---+
///
/// ... by merely specifying in any one node the snapshot at which it had transitioned to one or
/// the other, and the earliest snapshot in which it had already been reported.
#[derive(Debug)]
struct Node<O> {
    obligation: O,
    state: NodeState,
    parent: Option<NodeIndex>,
    root: NodeIndex, // points to the root, which may be the current node,
    scratch: NodeScratch,
}

/// Miscellaneous extra space for scratchwork while using the Node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeScratch {
    num_incomplete_children: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeSuccess {
    /// The number of incomplete children at the latest edit of the tree at `self.snapshot`.
    num_incomplete_children: usize,
    /// When this success was generated with some number of incomplete children.
    snapshot: Snapshot,
    /// When this success was reported (i.e. when its num_incomplete_children was noted to be 0).
    reported: Option<Snapshot>,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NodeErrorOrigin {
    Success(NodeSuccess),
    Pending,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeError {
    origin: NodeErrorOrigin,
    /// The snapshot from which this error originated (and was reported; this is a valid assumption
    /// due to the way the forest treats errors).
    snapshot: Snapshot,
}

/// The state of one node in some tree within the forest. This
/// represents the current state of processing for the obligation (of
/// type `O`) associated with this node.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
enum NodeState {
    /// Obligation not yet resolved to success or error.
    Pending,

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
    Success(NodeSuccess),

    /// This obligation was resolved to an error. Error nodes are
    /// removed from the vector by the compression step if they have
    /// no underlying snapshots that are still alive.
    Error(NodeError),
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
            snapshots: vec![Snapshot::new(0)]
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

    /// Get the current snapshot, initiating a new snapshot on top of it.
    pub fn start_snapshot(&mut self) -> Snapshot {
        let current_snapshot = self.current_snapshot();
        let next_snapshot = Snapshot::new(self.nodes.len());
        assert!(next_snapshot != current_snapshot);
        self.snapshots.push(next_snapshot.clone());
        current_snapshot
    }

    /// Commit to the given snapshot.
    pub fn commit_snapshot(&mut self, snapshot: Snapshot) {
        self.snapshots.pop();
        let prev_snapshot = self.current_snapshot();
        assert_eq!(prev_snapshot, snapshot);
        for node in &mut self.nodes {
            node.state.commit(prev_snapshot.clone());
        }
    }

    /// Rollback to the given snapshot.
    pub fn rollback_snapshot(&mut self, snapshot: Snapshot) {
        let nodes_len = self.snapshots.pop().unwrap().len;
        let prev_snapshot = self.current_snapshot();
        assert_eq!(prev_snapshot, snapshot);
        assert!(!self.snapshots.is_empty(), "rolled back into non-existence");
        self.nodes.truncate(nodes_len);
        for node in &mut self.nodes {
            node.state.rollback(prev_snapshot.clone())
        }
    }

    pub fn in_snapshot(&self) -> bool {
        // If we have 1 snapshot, we're at the base.
        self.snapshots.len() > 1
    }

    /// Adds a new tree to the forest.
    pub fn push_root(&mut self, obligation: O) {
        let index = NodeIndex::new(self.nodes.len());
        self.nodes.push(Node::new(obligation, NodeState::Pending, index, None));
    }

    /// Convert all remaining obligations to the given error.
    pub fn to_errors<E:Clone>(&mut self, error: E) -> Vec<Error<O,E>> {
        let current_snapshot = self.current_snapshot();
        let mut errors = vec![];
        for index in 0..self.nodes.len() {
            self.inherit_error(index);
            if self.nodes[index].state.is_pending(current_snapshot.clone()) {
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
        let current_snapshot = self.current_snapshot();
        self.nodes.iter()
                  .filter_map(|n| if n.state.is_pending(current_snapshot.clone()) {
                      Some(&n.obligation)
                  } else {
                      None
                  })
                  .cloned()
                  .collect()
    }

    /// Process the obligations.
    pub fn process_obligations<E,F>(&mut self, mut action: F) -> Outcome<O,E>
        where E: Debug, F: FnMut(&mut O, Backtrace<O>) -> Result<Option<Vec<O>>, E>
    {
        debug!("process_obligations(len={})", self.nodes.len());

        let current_snapshot = self.current_snapshot();

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
            if !self.nodes[index].state.is_pending(current_snapshot.clone()) {
                continue;
            }
            self.inherit_error(index);

            debug!("process_obligations: node {} == {:?}",
                   index, self.nodes[index].state);

            let result = {
                let parent = self.nodes[index].parent;
                let (prefix, suffix) = self.nodes.split_at_mut(index);
                let backtrace = Backtrace::new(prefix, parent, current_snapshot.clone());
                if suffix[0].state.is_pending(current_snapshot.clone()) {
                    action(&mut suffix[0].obligation, backtrace)
                } else {
                    continue;
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

        let current_snapshot = self.current_snapshot();
        let num_incomplete_children = children.len();

        if num_incomplete_children == 0 {
            // if there is no work left to be done, decrement parent's ref count
            self.update_parent(index);
        } else {
            // create child work
            let root_index = self.nodes[index].root;
            let node_index = NodeIndex::new(index);
            self.nodes.extend(
                children.into_iter()
                        .map(|o| Node::new(o,
                                           NodeState::Pending,
                                           root_index,
                                           Some(node_index))));
        }
        // change state from `Pending` to `Success`
        self.nodes[index].state.succeed(num_incomplete_children, current_snapshot);
    }

    /// Decrements the ref count on the parent of `child`; if the parent's ref count then reaches
    /// zero, proceeds recursively. Only updates counts for parents that are above the topmost
    /// snapshot length (the others have implicitly updated counts when reverse traversing the
    /// nodes).
    fn update_parent(&mut self, child: usize) {
        debug!("update_parent(child={})", child);
        let current_snapshot = self.current_snapshot();
        if let Some(parent) = self.nodes[child].parent {
            let parent = parent.get();
            if parent >= current_snapshot.len {
                if self.nodes[parent].state.decrement_incomplete_children(current_snapshot) ==
                    Some(0)
                {
                    self.update_parent(parent);
                }
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
        if self.nodes[root].state.is_error(current_snapshot.clone()) {
            self.nodes[child].state.error(current_snapshot);
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
            if self.nodes[p].state.is_error(current_snapshot.clone()) {
                panic!("encountered error in node {:?} when collecting stack trace", p);
            } else {
                trace.push(self.nodes[p].obligation.clone());
                self.nodes[p].state.error(current_snapshot.clone());
            }

            // loop to the parent
            match self.nodes[p].parent {
                Some(q) => { p = q.get(); }
                None => { return trace; }
            }
        }
    }

    /// Compresses the vector since the most recent snapshot, removing all popped nodes. This
    /// adjusts the indices and hence invalidates any outstanding indices. Only 'compresses' nodes
    /// above the most recent snapshot; always reports all successful nodes regardless of which
    /// snapshot they're in (as long as they haven't yet been reported in the current or preceding
    /// snapshots).
    fn compress(&mut self) -> Vec<O> {
        let current_snapshot = self.current_snapshot();
        // FIXME profile the effect of putting this in scratch (or using the scratch that's already
        // there) and make the appropriate updates.
        let mut rewrites: Vec<_> = (current_snapshot.len..self.nodes.len()).collect();

        // Finish propagating error state. Note that in this case we
        // only have to check immediate parents, rather than all
        // ancestors, because all errors have already occurred that
        // are going to occur.
        let nodes_len = self.nodes.len();
        for i in 0..nodes_len {
            if !self.nodes[i].state.is_done(current_snapshot.clone()) {
                self.inherit_error(i);
            }
        }

        // Now go through and move all nodes that are either
        // successful or which have an error over to the end of the
        // list, preserving the relative order of the survivors
        // (which is important for the `inherit_error` logic).
        let mut dead = 0;
        for i in current_snapshot.len..nodes_len {
            if self.nodes[i].state.is_done(current_snapshot.clone()) {
                dead += 1;
            } else if dead > 0 {
                self.nodes.swap(i, i - dead);
                rewrites[i - current_snapshot.len] -= dead;
            }
        }

        // Initialize scratch
        for node in (&mut self.nodes).into_iter().take(nodes_len - dead) {
            node.scratch.num_incomplete_children = match node.state {
                NodeState::Success(NodeSuccess { num_incomplete_children, .. }) =>
                    Some(num_incomplete_children),
                _ => None,
            };
        }

        // Initialize successful nodes return vec
        let mut successful = Vec::new();
        successful.reserve(nodes_len);

        // Pop off all the nodes we killed and extract the success
        // stories.
        successful.extend(
            (0 .. dead).map(|_| self.nodes.pop().unwrap())
                       .flat_map(|node| match node.state {
                           NodeState::Pending => unreachable!(),
                           NodeState::Error(_) => None,
                           NodeState::Success(_) => Some(node.obligation.clone()),
                       }));
        // Go through the still-alive successful and error nodes and extract the success stories,
        // marking them reported as we go along. The ordering here depends on the prefix property
        // of self.nodes.
        for i in (0..self.nodes.len()).rev() {
            if self.nodes[i].scratch.num_incomplete_children == Some(0) {
                assert!(self.nodes[i].state.is_success(current_snapshot.clone()));
                if let Some(parent) = self.nodes[i].parent {
                    assert!(parent.get() < i);
                    let _ = self.nodes[parent.get()].scratch.num_incomplete_children
                        .as_mut().map(|x| *x -= 1);
                }
                if !self.nodes[i].state.is_reported(current_snapshot.clone()) {
                    successful.push(self.nodes[i].obligation.clone());
                    self.nodes[i].state.report(current_snapshot.clone());
                }
            }
        }

        // Adjust the parent indices, since we compressed things.
        for node in (&mut self.nodes).into_iter().skip(current_snapshot.len) {
            if let Some(ref mut index) = node.parent {
                if index.get() >= current_snapshot.len {
                    let new_index = rewrites[index.get() - current_snapshot.len];
                    debug_assert!(new_index < (nodes_len - dead));
                    *index = NodeIndex::new(new_index);
                }
            }
            if node.root.get() >= current_snapshot.len {
                node.root = NodeIndex::new(rewrites[node.root.get() - current_snapshot.len]);
            }
        }
        successful
    }
}

#[derive(Clone)]
pub struct Backtrace<'b, O: 'b> {
    nodes: &'b [Node<O>],
    pointer: Option<NodeIndex>,
    snapshot: Snapshot,
}

impl<'b, O> Backtrace<'b, O> {
    fn new(nodes: &'b [Node<O>], pointer: Option<NodeIndex>, snapshot: Snapshot)
        -> Backtrace<'b, O>
    {
        Backtrace { nodes: nodes, pointer: pointer, snapshot: snapshot }
    }
}

impl<'b, O: Clone> Iterator for Backtrace<'b, O> {
    type Item = &'b O;

    fn next(&mut self) -> Option<&'b O> {
        debug!("Backtrace: self.pointer = {:?}", self.pointer);
        if let Some(p) = self.pointer {
            self.pointer = self.nodes[p.get()].parent;
            if self.nodes[p.get()].state.is_error(self.snapshot.clone()) {
                panic!("Backtrace encountered an error.")
            } else {
                Some(&self.nodes[p.get()].obligation)
            }
        } else {
            None
        }
    }
}
impl<O> Node<O> {
    fn new(obligation: O, state: NodeState, root: NodeIndex, parent: Option<NodeIndex>)
        -> Node<O>
    {
        Node {
            obligation: obligation,
            state: state,
            parent: parent,
            root: root,
            scratch: NodeScratch::default(),
        }
    }
}
impl NodeState {
    fn into_shim<F: FnOnce(Self) -> Self>(&mut self, transformer: F) {
        let new_self = transformer(mem::replace(self, unsafe { mem::uninitialized() } ));
        mem::forget(mem::replace(self, new_self));
    }
    fn succeed(&mut self, new_num_incomplete_children: usize, at_snapshot: Snapshot) {
        self.into_shim(|s| match s {
            NodeState::Pending => NodeState::Success(NodeSuccess {
                num_incomplete_children: new_num_incomplete_children,
                snapshot: at_snapshot,
                reported: None,
            }),
            NodeState::Error(_) |
            NodeState::Success(_) => panic!("invalid transition (non-pending to success)"),
        });
    }
    fn error(&mut self, at_snapshot: Snapshot) {
        self.into_shim(|s| match s {
            NodeState::Pending => NodeState::Error(NodeError {
                origin: NodeErrorOrigin::Pending,
                snapshot: at_snapshot,
            }),
            NodeState::Success(s) => if s.num_incomplete_children > 0 {
                assert!(s.snapshot <= at_snapshot, "invalid success-to-error snapshot ordering");
                assert!(s.reported.is_none(), "success should not have been reported if erroring");
                NodeState::Error(NodeError {
                    origin: NodeErrorOrigin::Success(s),
                    snapshot: at_snapshot,
                })
            } else {
                panic!("invalid transition (success with no children to error)")
            },
            NodeState::Error(_) => panic!("invalid transition (error to error)"),
        });
    }
    fn rollback(&mut self, to_snapshot: Snapshot) {
        self.into_shim(|s| match s {
            NodeState::Pending => NodeState::Pending,
            NodeState::Success(s) => {
                if s.snapshot <= to_snapshot {
                    NodeState::Success(NodeSuccess {
                        num_incomplete_children: s.num_incomplete_children,
                        snapshot: s.snapshot,
                        reported: match s.reported {
                            Some(reported_snapshot) => if reported_snapshot <= to_snapshot {
                                Some(reported_snapshot)
                            } else {
                                None
                            },
                            None => None
                        }
                    })
                } else {
                    NodeState::Pending
                }
            },
            NodeState::Error(NodeError { origin: NodeErrorOrigin::Pending, snapshot }) => {
                if snapshot <= to_snapshot {
                    NodeState::Error(NodeError {
                        origin: NodeErrorOrigin::Pending,
                        snapshot: snapshot
                    })
                } else {
                    NodeState::Pending
                }
            },
            NodeState::Error(NodeError { origin: NodeErrorOrigin::Success(s), snapshot }) => {
                if snapshot <= to_snapshot {
                    NodeState::Error(NodeError {
                        origin: NodeErrorOrigin::Success(s),
                        snapshot: snapshot
                    })
                } else {
                    let mut succ_roll = NodeState::Success(s);
                    succ_roll.rollback(to_snapshot);
                    succ_roll
                }
            },
        })
    }
    fn commit(&mut self, to_snapshot: Snapshot) {
        self.into_shim(|s| match s {
            NodeState::Pending => NodeState::Pending,
            NodeState::Success(s) => {
                NodeState::Success(NodeSuccess {
                    num_incomplete_children: s.num_incomplete_children,
                    snapshot: if to_snapshot < s.snapshot {
                        to_snapshot.clone()
                    } else {
                        s.snapshot
                    },
                    reported: match s.reported {
                        Some(reported_snapshot) => Some(if reported_snapshot <= to_snapshot {
                            reported_snapshot
                        } else {
                            to_snapshot
                        }),
                        None => None
                    },
                })
            },
            NodeState::Error(NodeError { origin: NodeErrorOrigin::Pending, snapshot }) => {
                NodeState::Error(NodeError {
                    origin: NodeErrorOrigin::Pending,
                    snapshot: if snapshot <= to_snapshot { snapshot } else { to_snapshot }
                })
            },
            NodeState::Error(NodeError { origin: NodeErrorOrigin::Success(s), snapshot }) => {
                let mut succ_roll = NodeState::Success(s);
                succ_roll.rollback(to_snapshot.clone());
                NodeState::Error(NodeError {
                    origin: match succ_roll {
                        NodeState::Pending => NodeErrorOrigin::Pending,
                        NodeState::Success(s) => NodeErrorOrigin::Success(s),
                        NodeState::Error(_) => unreachable!(),
                    },
                    snapshot: if snapshot <= to_snapshot { snapshot } else { to_snapshot }
                })
            },
        })
    }
    fn report(&mut self, at_snapshot: Snapshot) {
        self.into_shim(|s| match s {
            NodeState::Pending |
            NodeState::Error(_) => panic!("invalid transition (delayed reporting non-success)"),
            NodeState::Success(s) => {
                assert!(s.snapshot <= at_snapshot,
                        "cannot possibly report before having succeeded");
                NodeState::Success(NodeSuccess {
                    num_incomplete_children: s.num_incomplete_children,
                    snapshot: s.snapshot,
                    reported: match s.reported {
                        Some(_) => panic!("invalid transition (reporting already reported)"),
                        None => Some(at_snapshot),
                    },
                })
            },
        })
    }
    fn decrement_incomplete_children(&mut self, at_snapshot: Snapshot) -> Option<usize> {
        let mut result = None;
        self.into_shim(|s| match s {
            NodeState::Pending |
            NodeState::Error(_) => panic!("cannot decrement children of pending or error nodes"),
            NodeState::Success(s) => {
                assert!(s.snapshot <= at_snapshot,
                        "cannot possibly decrement outstanding children before having succeeded");
                result = Some(s.num_incomplete_children - 1);
                NodeState::Success(NodeSuccess {
                    num_incomplete_children: s.num_incomplete_children - 1,
                    snapshot: s.snapshot,
                    reported: s.reported,
                })
            },
        });
        result
    }
    /// Earliest snapshot at which the state transitioned to success.
    fn when_success(&self) -> Option<Snapshot> {
        match self {
            &NodeState::Pending => None,
            &NodeState::Success(NodeSuccess { ref snapshot, .. }) => Some(snapshot.clone()),
            &NodeState::Error(NodeError { origin: NodeErrorOrigin::Success(ref s) , .. }) =>
                Some(s.snapshot.clone()),
            &NodeState::Error(NodeError { origin: NodeErrorOrigin::Pending, .. }) => None,
        }
    }
    /// Earliest snapshot at which the state transitioned to error.
    fn when_error(&self) -> Option<Snapshot> {
        match self {
            &NodeState::Pending => None,
            &NodeState::Success(_) => None,
            &NodeState::Error(NodeError { ref snapshot, .. }) => Some(snapshot.clone()),
        }
    }
    /// Earliest snapshot at which the state transitioned to reported (either successful or error).
    fn when_reported(&self) -> Option<Snapshot> {
        match self {
            &NodeState::Pending => None,
            &NodeState::Success(NodeSuccess { reported: Some(ref snapshot), .. }) =>
                Some(snapshot.clone()),
            &NodeState::Success(NodeSuccess { reported: None, .. }) => None,
            &NodeState::Error(NodeError { ref snapshot, .. }) => Some(snapshot.clone()),
        }
    }
    fn is_pending(&self, at_snapshot: Snapshot) -> bool {
        self.when_success().map(|x| x > at_snapshot).unwrap_or(true) &&
            self.when_error().map(|x| x > at_snapshot).unwrap_or(true)
    }
    fn is_success(&self, at_snapshot: Snapshot) -> bool {
        self.when_success().map(|x| x <= at_snapshot).unwrap_or(false) &&
            self.when_error().map(|x| x > at_snapshot).unwrap_or(true)
    }
    fn is_reported(&self, at_snapshot: Snapshot) -> bool {
        self.when_reported().map(|x| x <= at_snapshot).unwrap_or(false)
    }
    fn is_childless_success(&self, at_snapshot: Snapshot) -> bool {
        // Because a success without children can never be an error, we just check with a match.
        match self {
            &NodeState::Success(NodeSuccess { num_incomplete_children: 0, ref snapshot, .. }) =>
                *snapshot <= at_snapshot,
            _ => false,
        }
    }
    fn is_error(&self, at_snapshot: Snapshot) -> bool {
        self.when_error().map(|x| at_snapshot >= x).unwrap_or(false)
    }
    fn is_done(&self, at_snapshot: Snapshot) -> bool {
        self.is_childless_success(at_snapshot.clone()) || self.is_error(at_snapshot)
    }
}
impl Default for NodeScratch {
    fn default() -> Self {
        NodeScratch { num_incomplete_children: None }
    }
}
impl Snapshot {
    fn new(len: usize) -> Self { Snapshot { len: len } }
}

