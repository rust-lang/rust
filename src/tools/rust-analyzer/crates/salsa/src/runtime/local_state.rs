use tracing::debug;
use triomphe::ThinArc;

use crate::durability::Durability;
use crate::runtime::ActiveQuery;
use crate::runtime::Revision;
use crate::Cycle;
use crate::DatabaseKeyIndex;
use std::cell::RefCell;

/// State that is specific to a single execution thread.
///
/// Internally, this type uses ref-cells.
///
/// **Note also that all mutations to the database handle (and hence
/// to the local-state) must be undone during unwinding.**
pub(super) struct LocalState {
    /// Vector of active queries.
    ///
    /// This is normally `Some`, but it is set to `None`
    /// while the query is blocked waiting for a result.
    ///
    /// Unwinding note: pushes onto this vector must be popped -- even
    /// during unwinding.
    query_stack: RefCell<Option<Vec<ActiveQuery>>>,
}

/// Summarizes "all the inputs that a query used"
#[derive(Debug, Clone)]
pub(crate) struct QueryRevisions {
    /// The most revision in which some input changed.
    pub(crate) changed_at: Revision,

    /// Minimum durability of the inputs to this query.
    pub(crate) durability: Durability,

    /// Whether the input is untracked.
    /// Invariant: if `untracked`, `inputs` is `None`.
    /// Why is this encoded like this and not a proper enum? Struct size, this saves us 8 bytes.
    pub(crate) untracked: bool,

    /// The inputs that went into our query, if we are tracking them.
    pub(crate) inputs: Option<ThinArc<(), DatabaseKeyIndex>>,
}

impl Default for LocalState {
    fn default() -> Self {
        LocalState { query_stack: RefCell::new(Some(Vec::new())) }
    }
}

impl LocalState {
    #[inline]
    pub(super) fn push_query(&self, database_key_index: DatabaseKeyIndex) -> ActiveQueryGuard<'_> {
        let mut query_stack = self.query_stack.borrow_mut();
        let query_stack = query_stack.as_mut().expect("local stack taken");
        query_stack.push(ActiveQuery::new(database_key_index));
        ActiveQueryGuard { local_state: self, database_key_index, push_len: query_stack.len() }
    }

    fn with_query_stack<R>(&self, c: impl FnOnce(&mut Vec<ActiveQuery>) -> R) -> R {
        c(self.query_stack.borrow_mut().as_mut().expect("query stack taken"))
    }

    pub(super) fn query_in_progress(&self) -> bool {
        self.with_query_stack(|stack| !stack.is_empty())
    }

    pub(super) fn active_query(&self) -> Option<DatabaseKeyIndex> {
        self.with_query_stack(|stack| {
            stack.last().map(|active_query| active_query.database_key_index)
        })
    }

    pub(super) fn report_query_read_and_unwind_if_cycle_resulted(
        &self,
        input: DatabaseKeyIndex,
        durability: Durability,
        changed_at: Revision,
    ) {
        debug!(
            "report_query_read_and_unwind_if_cycle_resulted(input={:?}, durability={:?}, changed_at={:?})",
            input, durability, changed_at
        );
        self.with_query_stack(|stack| {
            if let Some(top_query) = stack.last_mut() {
                top_query.add_read(input, durability, changed_at);

                // We are a cycle participant:
                //
                //     C0 --> ... --> Ci --> Ci+1 -> ... -> Cn --> C0
                //                        ^   ^
                //                        :   |
                //         This edge -----+   |
                //                            |
                //                            |
                //                            N0
                //
                // In this case, the value we have just read from `Ci+1`
                // is actually the cycle fallback value and not especially
                // interesting. We unwind now with `CycleParticipant` to avoid
                // executing the rest of our query function. This unwinding
                // will be caught and our own fallback value will be used.
                //
                // Note that `Ci+1` may` have *other* callers who are not
                // participants in the cycle (e.g., N0 in the graph above).
                // They will not have the `cycle` marker set in their
                // stack frames, so they will just read the fallback value
                // from `Ci+1` and continue on their merry way.
                if let Some(cycle) = &top_query.cycle {
                    cycle.clone().throw()
                }
            }
        })
    }

    pub(super) fn report_untracked_read(&self, current_revision: Revision) {
        self.with_query_stack(|stack| {
            if let Some(top_query) = stack.last_mut() {
                top_query.add_untracked_read(current_revision);
            }
        })
    }

    /// Update the top query on the stack to act as though it read a value
    /// of durability `durability` which changed in `revision`.
    pub(super) fn report_synthetic_read(&self, durability: Durability, revision: Revision) {
        self.with_query_stack(|stack| {
            if let Some(top_query) = stack.last_mut() {
                top_query.add_synthetic_read(durability, revision);
            }
        })
    }

    /// Takes the query stack and returns it. This is used when
    /// the current thread is blocking. The stack must be restored
    /// with [`Self::restore_query_stack`] when the thread unblocks.
    pub(super) fn take_query_stack(&self) -> Vec<ActiveQuery> {
        self.query_stack.take().expect("query stack already taken")
    }

    /// Restores a query stack taken with [`Self::take_query_stack`] once
    /// the thread unblocks.
    pub(super) fn restore_query_stack(&self, stack: Vec<ActiveQuery>) {
        assert!(self.query_stack.borrow().is_none(), "query stack not taken");
        self.query_stack.replace(Some(stack));
    }
}

impl std::panic::RefUnwindSafe for LocalState {}

/// When a query is pushed onto the `active_query` stack, this guard
/// is returned to represent its slot. The guard can be used to pop
/// the query from the stack -- in the case of unwinding, the guard's
/// destructor will also remove the query.
pub(crate) struct ActiveQueryGuard<'me> {
    local_state: &'me LocalState,
    push_len: usize,
    database_key_index: DatabaseKeyIndex,
}

impl ActiveQueryGuard<'_> {
    fn pop_helper(&self) -> ActiveQuery {
        self.local_state.with_query_stack(|stack| {
            // Sanity check: pushes and pops should be balanced.
            assert_eq!(stack.len(), self.push_len);
            debug_assert_eq!(stack.last().unwrap().database_key_index, self.database_key_index);
            stack.pop().unwrap()
        })
    }

    /// Invoked when the query has successfully completed execution.
    pub(super) fn complete(self) -> ActiveQuery {
        let query = self.pop_helper();
        std::mem::forget(self);
        query
    }

    /// Pops an active query from the stack. Returns the [`QueryRevisions`]
    /// which summarizes the other queries that were accessed during this
    /// query's execution.
    #[inline]
    pub(crate) fn pop(self) -> QueryRevisions {
        // Extract accumulated inputs.
        let popped_query = self.complete();

        // If this frame were a cycle participant, it would have unwound.
        assert!(popped_query.cycle.is_none());

        popped_query.revisions()
    }

    /// If the active query is registered as a cycle participant, remove and
    /// return that cycle.
    pub(crate) fn take_cycle(&self) -> Option<Cycle> {
        self.local_state.with_query_stack(|stack| stack.last_mut()?.cycle.take())
    }
}

impl Drop for ActiveQueryGuard<'_> {
    fn drop(&mut self) {
        self.pop_helper();
    }
}
