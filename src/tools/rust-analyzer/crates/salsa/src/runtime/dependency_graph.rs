use triomphe::Arc;

use crate::{DatabaseKeyIndex, RuntimeId};
use parking_lot::{Condvar, MutexGuard};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::{ActiveQuery, WaitResult};

type QueryStack = Vec<ActiveQuery>;

#[derive(Debug, Default)]
pub(super) struct DependencyGraph {
    /// A `(K -> V)` pair in this map indicates that the runtime
    /// `K` is blocked on some query executing in the runtime `V`.
    /// This encodes a graph that must be acyclic (or else deadlock
    /// will result).
    edges: FxHashMap<RuntimeId, Edge>,

    /// Encodes the `RuntimeId` that are blocked waiting for the result
    /// of a given query.
    query_dependents: FxHashMap<DatabaseKeyIndex, SmallVec<[RuntimeId; 4]>>,

    /// When a key K completes which had dependent queries Qs blocked on it,
    /// it stores its `WaitResult` here. As they wake up, each query Q in Qs will
    /// come here to fetch their results.
    wait_results: FxHashMap<RuntimeId, (QueryStack, WaitResult)>,
}

#[derive(Debug)]
struct Edge {
    blocked_on_id: RuntimeId,
    blocked_on_key: DatabaseKeyIndex,
    stack: QueryStack,

    /// Signalled whenever a query with dependents completes.
    /// Allows those dependents to check if they are ready to unblock.
    condvar: Arc<parking_lot::Condvar>,
}

impl DependencyGraph {
    /// True if `from_id` depends on `to_id`.
    ///
    /// (i.e., there is a path from `from_id` to `to_id` in the graph.)
    pub(super) fn depends_on(&mut self, from_id: RuntimeId, to_id: RuntimeId) -> bool {
        let mut p = from_id;
        while let Some(q) = self.edges.get(&p).map(|edge| edge.blocked_on_id) {
            if q == to_id {
                return true;
            }

            p = q;
        }
        p == to_id
    }

    /// Invokes `closure` with a `&mut ActiveQuery` for each query that participates in the cycle.
    /// The cycle runs as follows:
    ///
    /// 1. The runtime `from_id`, which has the stack `from_stack`, would like to invoke `database_key`...
    /// 2. ...but `database_key` is already being executed by `to_id`...
    /// 3. ...and `to_id` is transitively dependent on something which is present on `from_stack`.
    pub(super) fn for_each_cycle_participant(
        &mut self,
        from_id: RuntimeId,
        from_stack: &mut QueryStack,
        database_key: DatabaseKeyIndex,
        to_id: RuntimeId,
        mut closure: impl FnMut(&mut [ActiveQuery]),
    ) {
        debug_assert!(self.depends_on(to_id, from_id));

        // To understand this algorithm, consider this [drawing](https://is.gd/TGLI9v):
        //
        //    database_key = QB2
        //    from_id = A
        //    to_id = B
        //    from_stack = [QA1, QA2, QA3]
        //
        //    self.edges[B] = { C, QC2, [QB1..QB3] }
        //    self.edges[C] = { A, QA2, [QC1..QC3] }
        //
        //         The cyclic
        //         edge we have
        //         failed to add.
        //           :
        //    A      :    B         C
        //           :
        //    QA1    v    QB1       QC1
        // ┌► QA2    ┌──► QB2   ┌─► QC2
        // │  QA3 ───┘    QB3 ──┘   QC3 ───┐
        // │                               │
        // └───────────────────────────────┘
        //
        // Final output: [QB2, QB3, QC2, QC3, QA2, QA3]

        let mut id = to_id;
        let mut key = database_key;
        while id != from_id {
            // Looking at the diagram above, the idea is to
            // take the edge from `to_id` starting at `key`
            // (inclusive) and down to the end. We can then
            // load up the next thread (i.e., we start at B/QB2,
            // and then load up the dependency on C/QC2).
            let edge = self.edges.get_mut(&id).unwrap();
            let prefix = edge.stack.iter_mut().take_while(|p| p.database_key_index != key).count();
            closure(&mut edge.stack[prefix..]);
            id = edge.blocked_on_id;
            key = edge.blocked_on_key;
        }

        // Finally, we copy in the results from `from_stack`.
        let prefix = from_stack.iter_mut().take_while(|p| p.database_key_index != key).count();
        closure(&mut from_stack[prefix..]);
    }

    /// Unblock each blocked runtime (excluding the current one) if some
    /// query executing in that runtime is participating in cycle fallback.
    ///
    /// Returns a boolean (Current, Others) where:
    /// * Current is true if the current runtime has cycle participants
    ///   with fallback;
    /// * Others is true if other runtimes were unblocked.
    pub(super) fn maybe_unblock_runtimes_in_cycle(
        &mut self,
        from_id: RuntimeId,
        from_stack: &QueryStack,
        database_key: DatabaseKeyIndex,
        to_id: RuntimeId,
    ) -> (bool, bool) {
        // See diagram in `for_each_cycle_participant`.
        let mut id = to_id;
        let mut key = database_key;
        let mut others_unblocked = false;
        while id != from_id {
            let edge = self.edges.get(&id).unwrap();
            let prefix = edge.stack.iter().take_while(|p| p.database_key_index != key).count();
            let next_id = edge.blocked_on_id;
            let next_key = edge.blocked_on_key;

            if let Some(cycle) = edge.stack[prefix..].iter().rev().find_map(|aq| aq.cycle.clone()) {
                // Remove `id` from the list of runtimes blocked on `next_key`:
                self.query_dependents.get_mut(&next_key).unwrap().retain(|r| *r != id);

                // Unblock runtime so that it can resume execution once lock is released:
                self.unblock_runtime(id, WaitResult::Cycle(cycle));

                others_unblocked = true;
            }

            id = next_id;
            key = next_key;
        }

        let prefix = from_stack.iter().take_while(|p| p.database_key_index != key).count();
        let this_unblocked = from_stack[prefix..].iter().any(|aq| aq.cycle.is_some());

        (this_unblocked, others_unblocked)
    }

    /// Modifies the graph so that `from_id` is blocked
    /// on `database_key`, which is being computed by
    /// `to_id`.
    ///
    /// For this to be reasonable, the lock on the
    /// results table for `database_key` must be held.
    /// This ensures that computing `database_key` doesn't
    /// complete before `block_on` executes.
    ///
    /// Preconditions:
    /// * No path from `to_id` to `from_id`
    ///   (i.e., `me.depends_on(to_id, from_id)` is false)
    /// * `held_mutex` is a read lock (or stronger) on `database_key`
    pub(super) fn block_on<QueryMutexGuard>(
        mut me: MutexGuard<'_, Self>,
        from_id: RuntimeId,
        database_key: DatabaseKeyIndex,
        to_id: RuntimeId,
        from_stack: QueryStack,
        query_mutex_guard: QueryMutexGuard,
    ) -> (QueryStack, WaitResult) {
        let condvar = me.add_edge(from_id, database_key, to_id, from_stack);

        // Release the mutex that prevents `database_key`
        // from completing, now that the edge has been added.
        drop(query_mutex_guard);

        loop {
            if let Some(stack_and_result) = me.wait_results.remove(&from_id) {
                debug_assert!(!me.edges.contains_key(&from_id));
                return stack_and_result;
            }
            condvar.wait(&mut me);
        }
    }

    /// Helper for `block_on`: performs actual graph modification
    /// to add a dependency edge from `from_id` to `to_id`, which is
    /// computing `database_key`.
    fn add_edge(
        &mut self,
        from_id: RuntimeId,
        database_key: DatabaseKeyIndex,
        to_id: RuntimeId,
        from_stack: QueryStack,
    ) -> Arc<parking_lot::Condvar> {
        assert_ne!(from_id, to_id);
        debug_assert!(!self.edges.contains_key(&from_id));
        debug_assert!(!self.depends_on(to_id, from_id));

        let condvar = Arc::new(Condvar::new());
        self.edges.insert(
            from_id,
            Edge {
                blocked_on_id: to_id,
                blocked_on_key: database_key,
                stack: from_stack,
                condvar: condvar.clone(),
            },
        );
        self.query_dependents.entry(database_key).or_default().push(from_id);
        condvar
    }

    /// Invoked when runtime `to_id` completes executing
    /// `database_key`.
    pub(super) fn unblock_runtimes_blocked_on(
        &mut self,
        database_key: DatabaseKeyIndex,
        wait_result: WaitResult,
    ) {
        let dependents = self.query_dependents.remove(&database_key).unwrap_or_default();

        for from_id in dependents {
            self.unblock_runtime(from_id, wait_result.clone());
        }
    }

    /// Unblock the runtime with the given id with the given wait-result.
    /// This will cause it resume execution (though it will have to grab
    /// the lock on this data structure first, to recover the wait result).
    fn unblock_runtime(&mut self, id: RuntimeId, wait_result: WaitResult) {
        let edge = self.edges.remove(&id).expect("not blocked");
        self.wait_results.insert(id, (edge.stack, wait_result));

        // Now that we have inserted the `wait_results`,
        // notify the thread.
        edge.condvar.notify_one();
    }
}
