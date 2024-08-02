use crate::durability::Durability;
use crate::hash::FxIndexSet;
use crate::plumbing::CycleRecoveryStrategy;
use crate::revision::{AtomicRevision, Revision};
use crate::{Cancelled, Cycle, Database, DatabaseKeyIndex, Event, EventKind};
use itertools::Itertools;
use parking_lot::lock_api::{RawRwLock, RawRwLockRecursive};
use parking_lot::{Mutex, RwLock};
use std::hash::Hash;
use std::panic::panic_any;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::debug;
use triomphe::{Arc, ThinArc};

mod dependency_graph;
use dependency_graph::DependencyGraph;

pub(crate) mod local_state;
use local_state::LocalState;

use self::local_state::{ActiveQueryGuard, QueryRevisions};

/// The salsa runtime stores the storage for all queries as well as
/// tracking the query stack and dependencies between cycles.
///
/// Each new runtime you create (e.g., via `Runtime::new` or
/// `Runtime::default`) will have an independent set of query storage
/// associated with it. Normally, therefore, you only do this once, at
/// the start of your application.
pub struct Runtime {
    /// Our unique runtime id.
    id: RuntimeId,

    /// If this is a "forked" runtime, then the `revision_guard` will
    /// be `Some`; this guard holds a read-lock on the global query
    /// lock.
    revision_guard: Option<RevisionGuard>,

    /// Local state that is specific to this runtime (thread).
    local_state: LocalState,

    /// Shared state that is accessible via all runtimes.
    shared_state: Arc<SharedState>,
}

#[derive(Clone, Debug)]
pub(crate) enum WaitResult {
    Completed,
    Panicked,
    Cycle(Cycle),
}

impl Default for Runtime {
    fn default() -> Self {
        Runtime {
            id: RuntimeId { counter: 0 },
            revision_guard: None,
            shared_state: Default::default(),
            local_state: Default::default(),
        }
    }
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("Runtime")
            .field("id", &self.id())
            .field("forked", &self.revision_guard.is_some())
            .field("shared_state", &self.shared_state)
            .finish()
    }
}

impl Runtime {
    /// Create a new runtime; equivalent to `Self::default`. This is
    /// used when creating a new database.
    pub fn new() -> Self {
        Self::default()
    }

    /// See [`crate::storage::Storage::snapshot`].
    pub(crate) fn snapshot(&self) -> Self {
        if self.local_state.query_in_progress() {
            panic!("it is not legal to `snapshot` during a query (see salsa-rs/salsa#80)");
        }

        let revision_guard = RevisionGuard::new(&self.shared_state);

        let id = RuntimeId { counter: self.shared_state.next_id.fetch_add(1, Ordering::SeqCst) };

        Runtime {
            id,
            revision_guard: Some(revision_guard),
            shared_state: self.shared_state.clone(),
            local_state: Default::default(),
        }
    }

    /// A "synthetic write" causes the system to act *as though* some
    /// input of durability `durability` has changed. This is mostly
    /// useful for profiling scenarios.
    ///
    /// **WARNING:** Just like an ordinary write, this method triggers
    /// cancellation. If you invoke it while a snapshot exists, it
    /// will block until that snapshot is dropped -- if that snapshot
    /// is owned by the current thread, this could trigger deadlock.
    pub fn synthetic_write(&mut self, durability: Durability) {
        self.with_incremented_revision(|_next_revision| Some(durability));
    }

    /// The unique identifier attached to this `SalsaRuntime`. Each
    /// snapshotted runtime has a distinct identifier.
    #[inline]
    pub fn id(&self) -> RuntimeId {
        self.id
    }

    /// Returns the database-key for the query that this thread is
    /// actively executing (if any).
    pub fn active_query(&self) -> Option<DatabaseKeyIndex> {
        self.local_state.active_query()
    }

    /// Read current value of the revision counter.
    #[inline]
    pub(crate) fn current_revision(&self) -> Revision {
        self.shared_state.revisions[0].load()
    }

    /// The revision in which values with durability `d` may have last
    /// changed.  For D0, this is just the current revision. But for
    /// higher levels of durability, this value may lag behind the
    /// current revision. If we encounter a value of durability Di,
    /// then, we can check this function to get a "bound" on when the
    /// value may have changed, which allows us to skip walking its
    /// dependencies.
    #[inline]
    pub(crate) fn last_changed_revision(&self, d: Durability) -> Revision {
        self.shared_state.revisions[d.index()].load()
    }

    /// Read current value of the revision counter.
    #[inline]
    pub(crate) fn pending_revision(&self) -> Revision {
        self.shared_state.pending_revision.load()
    }

    #[cold]
    pub(crate) fn unwind_cancelled(&self) {
        self.report_untracked_read();
        Cancelled::PendingWrite.throw();
    }

    /// Acquires the **global query write lock** (ensuring that no queries are
    /// executing) and then increments the current revision counter; invokes
    /// `op` with the global query write lock still held.
    ///
    /// While we wait to acquire the global query write lock, this method will
    /// also increment `pending_revision_increments`, thus signalling to queries
    /// that their results are "cancelled" and they should abort as expeditiously
    /// as possible.
    ///
    /// The `op` closure should actually perform the writes needed. It is given
    /// the new revision as an argument, and its return value indicates whether
    /// any pre-existing value was modified:
    ///
    /// - returning `None` means that no pre-existing value was modified (this
    ///   could occur e.g. when setting some key on an input that was never set
    ///   before)
    /// - returning `Some(d)` indicates that a pre-existing value was modified
    ///   and it had the durability `d`. This will update the records for when
    ///   values with each durability were modified.
    ///
    /// Note that, given our writer model, we can assume that only one thread is
    /// attempting to increment the global revision at a time.
    pub(crate) fn with_incremented_revision<F>(&mut self, op: F)
    where
        F: FnOnce(Revision) -> Option<Durability>,
    {
        tracing::debug!("increment_revision()");

        if !self.permits_increment() {
            panic!("increment_revision invoked during a query computation");
        }

        // Set the `pending_revision` field so that people
        // know current revision is cancelled.
        let current_revision = self.shared_state.pending_revision.fetch_then_increment();

        // To modify the revision, we need the lock.
        let shared_state = self.shared_state.clone();
        let _lock = shared_state.query_lock.write();

        let old_revision = self.shared_state.revisions[0].fetch_then_increment();
        assert_eq!(current_revision, old_revision);

        let new_revision = current_revision.next();

        debug!("increment_revision: incremented to {:?}", new_revision);

        if let Some(d) = op(new_revision) {
            for rev in &self.shared_state.revisions[1..=d.index()] {
                rev.store(new_revision);
            }
        }
    }

    pub(crate) fn permits_increment(&self) -> bool {
        self.revision_guard.is_none() && !self.local_state.query_in_progress()
    }

    #[inline]
    pub(crate) fn push_query(&self, database_key_index: DatabaseKeyIndex) -> ActiveQueryGuard<'_> {
        self.local_state.push_query(database_key_index)
    }

    /// Reports that the currently active query read the result from
    /// another query.
    ///
    /// Also checks whether the "cycle participant" flag is set on
    /// the current stack frame -- if so, panics with `CycleParticipant`
    /// value, which should be caught by the code executing the query.
    ///
    /// # Parameters
    ///
    /// - `database_key`: the query whose result was read
    /// - `changed_revision`: the last revision in which the result of that
    ///   query had changed
    pub(crate) fn report_query_read_and_unwind_if_cycle_resulted(
        &self,
        input: DatabaseKeyIndex,
        durability: Durability,
        changed_at: Revision,
    ) {
        self.local_state
            .report_query_read_and_unwind_if_cycle_resulted(input, durability, changed_at);
    }

    /// Reports that the query depends on some state unknown to salsa.
    ///
    /// Queries which report untracked reads will be re-executed in the next
    /// revision.
    pub fn report_untracked_read(&self) {
        self.local_state.report_untracked_read(self.current_revision());
    }

    /// Acts as though the current query had read an input with the given durability; this will force the current query's durability to be at most `durability`.
    ///
    /// This is mostly useful to control the durability level for [on-demand inputs](https://salsa-rs.github.io/salsa/common_patterns/on_demand_inputs.html).
    pub fn report_synthetic_read(&self, durability: Durability) {
        let changed_at = self.last_changed_revision(durability);
        self.local_state.report_synthetic_read(durability, changed_at);
    }

    /// Handles a cycle in the dependency graph that was detected when the
    /// current thread tried to block on `database_key_index` which is being
    /// executed by `to_id`. If this function returns, then `to_id` no longer
    /// depends on the current thread, and so we should continue executing
    /// as normal. Otherwise, the function will throw a `Cycle` which is expected
    /// to be caught by some frame on our stack. This occurs either if there is
    /// a frame on our stack with cycle recovery (possibly the top one!) or if there
    /// is no cycle recovery at all.
    fn unblock_cycle_and_maybe_throw(
        &self,
        db: &dyn Database,
        dg: &mut DependencyGraph,
        database_key_index: DatabaseKeyIndex,
        to_id: RuntimeId,
    ) {
        debug!("unblock_cycle_and_maybe_throw(database_key={:?})", database_key_index);

        let mut from_stack = self.local_state.take_query_stack();
        let from_id = self.id();

        // Make a "dummy stack frame". As we iterate through the cycle, we will collect the
        // inputs from each participant. Then, if we are participating in cycle recovery, we
        // will propagate those results to all participants.
        let mut cycle_query = ActiveQuery::new(database_key_index);

        // Identify the cycle participants:
        let cycle = {
            let mut v = vec![];
            dg.for_each_cycle_participant(
                from_id,
                &mut from_stack,
                database_key_index,
                to_id,
                |aqs| {
                    aqs.iter_mut().for_each(|aq| {
                        cycle_query.add_from(aq);
                        v.push(aq.database_key_index);
                    });
                },
            );

            // We want to give the participants in a deterministic order
            // (at least for this execution, not necessarily across executions),
            // no matter where it started on the stack. Find the minimum
            // key and rotate it to the front.
            let index = v.iter().position_min().unwrap_or_default();
            v.rotate_left(index);

            // No need to store extra memory.
            v.shrink_to_fit();

            Cycle::new(Arc::new(v))
        };
        debug!("cycle {:?}, cycle_query {:#?}", cycle.debug(db), cycle_query,);

        // We can remove the cycle participants from the list of dependencies;
        // they are a strongly connected component (SCC) and we only care about
        // dependencies to things outside the SCC that control whether it will
        // form again.
        cycle_query.remove_cycle_participants(&cycle);

        // Mark each cycle participant that has recovery set, along with
        // any frames that come after them on the same thread. Those frames
        // are going to be unwound so that fallback can occur.
        dg.for_each_cycle_participant(from_id, &mut from_stack, database_key_index, to_id, |aqs| {
            aqs.iter_mut()
                .skip_while(|aq| match db.cycle_recovery_strategy(aq.database_key_index) {
                    CycleRecoveryStrategy::Panic => true,
                    CycleRecoveryStrategy::Fallback => false,
                })
                .for_each(|aq| {
                    debug!("marking {:?} for fallback", aq.database_key_index.debug(db));
                    aq.take_inputs_from(&cycle_query);
                    assert!(aq.cycle.is_none());
                    aq.cycle = Some(cycle.clone());
                });
        });

        // Unblock every thread that has cycle recovery with a `WaitResult::Cycle`.
        // They will throw the cycle, which will be caught by the frame that has
        // cycle recovery so that it can execute that recovery.
        let (me_recovered, others_recovered) =
            dg.maybe_unblock_runtimes_in_cycle(from_id, &from_stack, database_key_index, to_id);

        self.local_state.restore_query_stack(from_stack);

        if me_recovered {
            // If the current thread has recovery, we want to throw
            // so that it can begin.
            cycle.throw()
        } else if others_recovered {
            // If other threads have recovery but we didn't: return and we will block on them.
        } else {
            // if nobody has recover, then we panic
            panic_any(cycle);
        }
    }

    /// Block until `other_id` completes executing `database_key`;
    /// panic or unwind in the case of a cycle.
    ///
    /// `query_mutex_guard` is the guard for the current query's state;
    /// it will be dropped after we have successfully registered the
    /// dependency.
    ///
    /// # Propagating panics
    ///
    /// If the thread `other_id` panics, then our thread is considered
    /// cancelled, so this function will panic with a `Cancelled` value.
    ///
    /// # Cycle handling
    ///
    /// If the thread `other_id` already depends on the current thread,
    /// and hence there is a cycle in the query graph, then this function
    /// will unwind instead of returning normally. The method of unwinding
    /// depends on the [`Self::mutual_cycle_recovery_strategy`]
    /// of the cycle participants:
    ///
    /// * [`CycleRecoveryStrategy::Panic`]: panic with the [`Cycle`] as the value.
    /// * [`CycleRecoveryStrategy::Fallback`]: initiate unwinding with [`CycleParticipant::unwind`].
    pub(crate) fn block_on_or_unwind<QueryMutexGuard>(
        &self,
        db: &dyn Database,
        database_key: DatabaseKeyIndex,
        other_id: RuntimeId,
        query_mutex_guard: QueryMutexGuard,
    ) {
        let mut dg = self.shared_state.dependency_graph.lock();

        if dg.depends_on(other_id, self.id()) {
            self.unblock_cycle_and_maybe_throw(db, &mut dg, database_key, other_id);

            // If the above fn returns, then (via cycle recovery) it has unblocked the
            // cycle, so we can continue.
            assert!(!dg.depends_on(other_id, self.id()));
        }

        db.salsa_event(Event {
            runtime_id: self.id(),
            kind: EventKind::WillBlockOn { other_runtime_id: other_id, database_key },
        });

        let stack = self.local_state.take_query_stack();

        let (stack, result) = DependencyGraph::block_on(
            dg,
            self.id(),
            database_key,
            other_id,
            stack,
            query_mutex_guard,
        );

        self.local_state.restore_query_stack(stack);

        match result {
            WaitResult::Completed => (),

            // If the other thread panicked, then we consider this thread
            // cancelled. The assumption is that the panic will be detected
            // by the other thread and responded to appropriately.
            WaitResult::Panicked => Cancelled::PropagatedPanic.throw(),

            WaitResult::Cycle(c) => c.throw(),
        }
    }

    /// Invoked when this runtime completed computing `database_key` with
    /// the given result `wait_result` (`wait_result` should be `None` if
    /// computing `database_key` panicked and could not complete).
    /// This function unblocks any dependent queries and allows them
    /// to continue executing.
    pub(crate) fn unblock_queries_blocked_on(
        &self,
        database_key: DatabaseKeyIndex,
        wait_result: WaitResult,
    ) {
        self.shared_state
            .dependency_graph
            .lock()
            .unblock_runtimes_blocked_on(database_key, wait_result);
    }
}

/// State that will be common to all threads (when we support multiple threads)
struct SharedState {
    /// Stores the next id to use for a snapshotted runtime (starts at 1).
    next_id: AtomicU32,

    /// Whenever derived queries are executing, they acquire this lock
    /// in read mode. Mutating inputs (and thus creating a new
    /// revision) requires a write lock (thus guaranteeing that no
    /// derived queries are in progress). Note that this is not needed
    /// to prevent **race conditions** -- the revision counter itself
    /// is stored in an `AtomicUsize` so it can be cheaply read
    /// without acquiring the lock.  Rather, the `query_lock` is used
    /// to ensure a higher-level consistency property.
    query_lock: RwLock<()>,

    /// This is typically equal to `revision` -- set to `revision+1`
    /// when a new revision is pending (which implies that the current
    /// revision is cancelled).
    pending_revision: AtomicRevision,

    /// Stores the "last change" revision for values of each Durability.
    /// This vector is always of length at least 1 (for Durability 0)
    /// but its total length depends on the number of Durabilities. The
    /// element at index 0 is special as it represents the "current
    /// revision".  In general, we have the invariant that revisions
    /// in here are *declining* -- that is, `revisions[i] >=
    /// revisions[i + 1]`, for all `i`. This is because when you
    /// modify a value with durability D, that implies that values
    /// with durability less than D may have changed too.
    revisions: [AtomicRevision; Durability::LEN],

    /// The dependency graph tracks which runtimes are blocked on one
    /// another, waiting for queries to terminate.
    dependency_graph: Mutex<DependencyGraph>,
}

impl std::panic::RefUnwindSafe for SharedState {}

impl Default for SharedState {
    fn default() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const START: AtomicRevision = AtomicRevision::start();
        SharedState {
            next_id: AtomicU32::new(1),
            query_lock: Default::default(),
            revisions: [START; Durability::LEN],
            pending_revision: START,
            dependency_graph: Default::default(),
        }
    }
}

impl std::fmt::Debug for SharedState {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let query_lock = if self.query_lock.is_locked_exclusive() {
            "<wlocked>"
        } else if self.query_lock.is_locked() {
            "<rlocked>"
        } else {
            "<unlocked>"
        };
        fmt.debug_struct("SharedState")
            .field("query_lock", &query_lock)
            .field("revisions", &self.revisions)
            .field("pending_revision", &self.pending_revision)
            .finish()
    }
}

#[derive(Debug)]
struct ActiveQuery {
    /// What query is executing
    database_key_index: DatabaseKeyIndex,

    /// Minimum durability of inputs observed so far.
    durability: Durability,

    /// Maximum revision of all inputs observed. If we observe an
    /// untracked read, this will be set to the most recent revision.
    changed_at: Revision,

    /// Set of subqueries that were accessed thus far, or `None` if
    /// there was an untracked the read.
    dependencies: Option<FxIndexSet<DatabaseKeyIndex>>,

    /// Stores the entire cycle, if one is found and this query is part of it.
    cycle: Option<Cycle>,
}

impl ActiveQuery {
    fn new(database_key_index: DatabaseKeyIndex) -> Self {
        ActiveQuery {
            database_key_index,
            durability: Durability::MAX,
            changed_at: Revision::start(),
            dependencies: Some(FxIndexSet::default()),
            cycle: None,
        }
    }

    fn add_read(&mut self, input: DatabaseKeyIndex, durability: Durability, revision: Revision) {
        if let Some(set) = &mut self.dependencies {
            set.insert(input);
        }

        self.durability = self.durability.min(durability);
        self.changed_at = self.changed_at.max(revision);
    }

    fn add_untracked_read(&mut self, changed_at: Revision) {
        self.dependencies = None;
        self.durability = Durability::LOW;
        self.changed_at = changed_at;
    }

    fn add_synthetic_read(&mut self, durability: Durability, revision: Revision) {
        self.dependencies = None;
        self.durability = self.durability.min(durability);
        self.changed_at = self.changed_at.max(revision);
    }

    pub(crate) fn revisions(&self) -> QueryRevisions {
        let (inputs, untracked) = match &self.dependencies {
            None => (None, true),

            Some(dependencies) => (
                if dependencies.is_empty() {
                    None
                } else {
                    Some(ThinArc::from_header_and_iter((), dependencies.iter().copied()))
                },
                false,
            ),
        };

        QueryRevisions {
            changed_at: self.changed_at,
            inputs,
            untracked,
            durability: self.durability,
        }
    }

    /// Adds any dependencies from `other` into `self`.
    /// Used during cycle recovery, see [`Runtime::create_cycle_error`].
    fn add_from(&mut self, other: &ActiveQuery) {
        self.changed_at = self.changed_at.max(other.changed_at);
        self.durability = self.durability.min(other.durability);
        if let Some(other_dependencies) = &other.dependencies {
            if let Some(my_dependencies) = &mut self.dependencies {
                my_dependencies.extend(other_dependencies.iter().copied());
            }
        } else {
            self.dependencies = None;
        }
    }

    /// Removes the participants in `cycle` from my dependencies.
    /// Used during cycle recovery, see [`Runtime::create_cycle_error`].
    fn remove_cycle_participants(&mut self, cycle: &Cycle) {
        if let Some(my_dependencies) = &mut self.dependencies {
            for p in cycle.participant_keys() {
                my_dependencies.swap_remove(&p);
            }
        }
    }

    /// Copy the changed-at, durability, and dependencies from `cycle_query`.
    /// Used during cycle recovery, see [`Runtime::create_cycle_error`].
    pub(crate) fn take_inputs_from(&mut self, cycle_query: &ActiveQuery) {
        self.changed_at = cycle_query.changed_at;
        self.durability = cycle_query.durability;
        self.dependencies.clone_from(&cycle_query.dependencies);
    }
}

/// A unique identifier for a particular runtime. Each time you create
/// a snapshot, a fresh `RuntimeId` is generated. Once a snapshot is
/// complete, its `RuntimeId` may potentially be re-used.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuntimeId {
    counter: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct StampedValue<V> {
    pub(crate) value: V,
    pub(crate) durability: Durability,
    pub(crate) changed_at: Revision,
}

struct RevisionGuard {
    shared_state: Arc<SharedState>,
}

impl RevisionGuard {
    fn new(shared_state: &Arc<SharedState>) -> Self {
        // Subtle: we use a "recursive" lock here so that it is not an
        // error to acquire a read-lock when one is already held (this
        // happens when a query uses `snapshot` to spawn off parallel
        // workers, for example).
        //
        // This has the side-effect that we are responsible to ensure
        // that people contending for the write lock do not starve,
        // but this is what we achieve via the cancellation mechanism.
        //
        // (In particular, since we only ever have one "mutating
        // handle" to the database, the only contention for the global
        // query lock occurs when there are "futures" evaluating
        // queries in parallel, and those futures hold a read-lock
        // already, so the starvation problem is more about them bring
        // themselves to a close, versus preventing other people from
        // *starting* work).
        unsafe {
            shared_state.query_lock.raw().lock_shared_recursive();
        }

        Self { shared_state: shared_state.clone() }
    }
}

impl Drop for RevisionGuard {
    fn drop(&mut self) {
        // Release our read-lock without using RAII. As documented in
        // `Snapshot::new` above, this requires the unsafe keyword.
        unsafe {
            self.shared_state.query_lock.raw().unlock_shared();
        }
    }
}
