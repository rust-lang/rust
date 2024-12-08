use crate::debug::TableEntry;
use crate::durability::Durability;
use crate::plumbing::{DatabaseOps, QueryFunction};
use crate::revision::Revision;
use crate::runtime::local_state::ActiveQueryGuard;
use crate::runtime::local_state::QueryRevisions;
use crate::runtime::Runtime;
use crate::runtime::RuntimeId;
use crate::runtime::StampedValue;
use crate::runtime::WaitResult;
use crate::Cycle;
use crate::{Database, DatabaseKeyIndex, Event, EventKind, QueryDb};
use parking_lot::{RawRwLock, RwLock};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info};

pub(super) struct Slot<Q>
where
    Q: QueryFunction,
{
    key_index: u32,
    // FIXME: Yeet this
    group_index: u16,
    state: RwLock<QueryState<Q>>,
}

/// Defines the "current state" of query's memoized results.
enum QueryState<Q>
where
    Q: QueryFunction,
{
    NotComputed,

    /// The runtime with the given id is currently computing the
    /// result of this query.
    InProgress {
        id: RuntimeId,

        /// Set to true if any other queries are blocked,
        /// waiting for this query to complete.
        anyone_waiting: AtomicBool,
    },

    /// We have computed the query already, and here is the result.
    Memoized(Memo<Q::Value>),
}

struct Memo<V> {
    /// The result of the query, if we decide to memoize it.
    value: V,

    /// Last revision when this memo was verified; this begins
    /// as the current revision.
    pub(crate) verified_at: Revision,

    /// Revision information
    revisions: QueryRevisions,
}

/// Return value of `probe` helper.
enum ProbeState<V, G> {
    /// Another thread was active but has completed.
    /// Try again!
    Retry,

    /// No entry for this key at all.
    NotComputed(G),

    /// There is an entry, but its contents have not been
    /// verified in this revision.
    Stale(G),

    /// There is an entry which has been verified,
    /// and it has the following value-- or, we blocked
    /// on another thread, and that resulted in a cycle.
    UpToDate(V),
}

/// Return value of `maybe_changed_after_probe` helper.
enum MaybeChangedSinceProbeState<G> {
    /// Another thread was active but has completed.
    /// Try again!
    Retry,

    /// Value may have changed in the given revision.
    ChangedAt(Revision),

    /// There is a stale cache entry that has not been
    /// verified in this revision, so we can't say.
    Stale(G),
}

impl<Q> Slot<Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    pub(super) fn new(database_key_index: DatabaseKeyIndex) -> Self {
        Self {
            key_index: database_key_index.key_index,
            group_index: database_key_index.group_index,
            state: RwLock::new(QueryState::NotComputed),
        }
    }

    pub(super) fn database_key_index(&self) -> DatabaseKeyIndex {
        DatabaseKeyIndex {
            group_index: self.group_index,
            query_index: Q::QUERY_INDEX,
            key_index: self.key_index,
        }
    }

    pub(super) fn read(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        key: &Q::Key,
    ) -> StampedValue<Q::Value> {
        let runtime = db.salsa_runtime();

        // NB: We don't need to worry about people modifying the
        // revision out from under our feet. Either `db` is a frozen
        // database, in which case there is a lock, or the mutator
        // thread is the current thread, and it will be prevented from
        // doing any `set` invocations while the query function runs.
        let revision_now = runtime.current_revision();

        info!("{:?}: invoked at {:?}", self, revision_now,);

        // First, do a check with a read-lock.
        loop {
            match self.probe(db, self.state.read(), runtime, revision_now) {
                ProbeState::UpToDate(v) => return v,
                ProbeState::Stale(..) | ProbeState::NotComputed(..) => break,
                ProbeState::Retry => continue,
            }
        }

        self.read_upgrade(db, key, revision_now)
    }

    /// Second phase of a read operation: acquires an upgradable-read
    /// and -- if needed -- validates whether inputs have changed,
    /// recomputes value, etc. This is invoked after our initial probe
    /// shows a potentially out of date value.
    fn read_upgrade(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        key: &Q::Key,
        revision_now: Revision,
    ) -> StampedValue<Q::Value> {
        let runtime = db.salsa_runtime();

        debug!("{:?}: read_upgrade(revision_now={:?})", self, revision_now,);

        // Check with an upgradable read to see if there is a value
        // already. (This permits other readers but prevents anyone
        // else from running `read_upgrade` at the same time.)
        let mut old_memo = loop {
            match self.probe(db, self.state.upgradable_read(), runtime, revision_now) {
                ProbeState::UpToDate(v) => return v,
                ProbeState::Stale(state) | ProbeState::NotComputed(state) => {
                    type RwLockUpgradableReadGuard<'a, T> =
                        lock_api::RwLockUpgradableReadGuard<'a, RawRwLock, T>;

                    let mut state = RwLockUpgradableReadGuard::upgrade(state);
                    match std::mem::replace(&mut *state, QueryState::in_progress(runtime.id())) {
                        QueryState::Memoized(old_memo) => break Some(old_memo),
                        QueryState::InProgress { .. } => unreachable!(),
                        QueryState::NotComputed => break None,
                    }
                }
                ProbeState::Retry => continue,
            }
        };

        let panic_guard = PanicGuard::new(self, runtime);
        let active_query = runtime.push_query(self.database_key_index());

        // If we have an old-value, it *may* now be stale, since there
        // has been a new revision since the last time we checked. So,
        // first things first, let's walk over each of our previous
        // inputs and check whether they are out of date.
        if let Some(memo) = &mut old_memo {
            if let Some(value) = memo.verify_value(db.ops_database(), revision_now, &active_query) {
                info!("{:?}: validated old memoized value", self,);

                db.salsa_event(Event {
                    runtime_id: runtime.id(),
                    kind: EventKind::DidValidateMemoizedValue {
                        database_key: self.database_key_index(),
                    },
                });

                panic_guard.proceed(old_memo);

                return value;
            }
        }

        self.execute(db, runtime, revision_now, active_query, panic_guard, old_memo, key)
    }

    fn execute(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        runtime: &Runtime,
        revision_now: Revision,
        active_query: ActiveQueryGuard<'_>,
        panic_guard: PanicGuard<'_, Q>,
        old_memo: Option<Memo<Q::Value>>,
        key: &Q::Key,
    ) -> StampedValue<Q::Value> {
        tracing::info!("{:?}: executing query", self.database_key_index().debug(db));

        db.salsa_event(Event {
            runtime_id: db.salsa_runtime().id(),
            kind: EventKind::WillExecute { database_key: self.database_key_index() },
        });

        // Query was not previously executed, or value is potentially
        // stale, or value is absent. Let's execute!
        let value = match Cycle::catch(|| Q::execute(db, key.clone())) {
            Ok(v) => v,
            Err(cycle) => {
                tracing::debug!(
                    "{:?}: caught cycle {:?}, have strategy {:?}",
                    self.database_key_index().debug(db),
                    cycle,
                    Q::CYCLE_STRATEGY,
                );
                match Q::CYCLE_STRATEGY {
                    crate::plumbing::CycleRecoveryStrategy::Panic => {
                        panic_guard.proceed(None);
                        cycle.throw()
                    }
                    crate::plumbing::CycleRecoveryStrategy::Fallback => {
                        if let Some(c) = active_query.take_cycle() {
                            assert!(c.is(&cycle));
                            Q::cycle_fallback(db, &cycle, key)
                        } else {
                            // we are not a participant in this cycle
                            debug_assert!(!cycle
                                .participant_keys()
                                .any(|k| k == self.database_key_index()));
                            cycle.throw()
                        }
                    }
                }
            }
        };

        let mut revisions = active_query.pop();

        // We assume that query is side-effect free -- that is, does
        // not mutate the "inputs" to the query system. Sanity check
        // that assumption here, at least to the best of our ability.
        assert_eq!(
            runtime.current_revision(),
            revision_now,
            "revision altered during query execution",
        );

        // If the new value is equal to the old one, then it didn't
        // really change, even if some of its inputs have. So we can
        // "backdate" its `changed_at` revision to be the same as the
        // old value.
        if let Some(old_memo) = &old_memo {
            // Careful: if the value became less durable than it
            // used to be, that is a "breaking change" that our
            // consumers must be aware of. Becoming *more* durable
            // is not. See the test `constant_to_non_constant`.
            if revisions.durability >= old_memo.revisions.durability && old_memo.value == value {
                debug!(
                    "read_upgrade({:?}): value is equal, back-dating to {:?}",
                    self, old_memo.revisions.changed_at,
                );

                assert!(old_memo.revisions.changed_at <= revisions.changed_at);
                revisions.changed_at = old_memo.revisions.changed_at;
            }
        }

        let new_value = StampedValue {
            value,
            durability: revisions.durability,
            changed_at: revisions.changed_at,
        };

        let memo_value = new_value.value.clone();

        debug!("read_upgrade({:?}): result.revisions = {:#?}", self, revisions,);

        panic_guard.proceed(Some(Memo { value: memo_value, verified_at: revision_now, revisions }));

        new_value
    }

    /// Helper for `read` that does a shallow check (not recursive) if we have an up-to-date value.
    ///
    /// Invoked with the guard `state` corresponding to the `QueryState` of some `Slot` (the guard
    /// can be either read or write). Returns a suitable `ProbeState`:
    ///
    /// - `ProbeState::UpToDate(r)` if the table has an up-to-date value (or we blocked on another
    ///   thread that produced such a value).
    /// - `ProbeState::StaleOrAbsent(g)` if either (a) there is no memo for this key, (b) the memo
    ///   has no value; or (c) the memo has not been verified at the current revision.
    ///
    /// Note that in case `ProbeState::UpToDate`, the lock will have been released.
    fn probe<StateGuard>(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        state: StateGuard,
        runtime: &Runtime,
        revision_now: Revision,
    ) -> ProbeState<StampedValue<Q::Value>, StateGuard>
    where
        StateGuard: Deref<Target = QueryState<Q>>,
    {
        match &*state {
            QueryState::NotComputed => ProbeState::NotComputed(state),

            QueryState::InProgress { id, anyone_waiting } => {
                let other_id = *id;

                // NB: `Ordering::Relaxed` is sufficient here,
                // as there are no loads that are "gated" on this
                // value. Everything that is written is also protected
                // by a lock that must be acquired. The role of this
                // boolean is to decide *whether* to acquire the lock,
                // not to gate future atomic reads.
                anyone_waiting.store(true, Ordering::Relaxed);

                self.block_on_or_unwind(db, runtime, other_id, state);

                // Other thread completely normally, so our value may be available now.
                ProbeState::Retry
            }

            QueryState::Memoized(memo) => {
                debug!(
                    "{:?}: found memoized value, verified_at={:?}, changed_at={:?}",
                    self, memo.verified_at, memo.revisions.changed_at,
                );

                if memo.verified_at < revision_now {
                    return ProbeState::Stale(state);
                }

                let value = &memo.value;
                let value = StampedValue {
                    durability: memo.revisions.durability,
                    changed_at: memo.revisions.changed_at,
                    value: value.clone(),
                };

                info!("{:?}: returning memoized value changed at {:?}", self, value.changed_at);

                ProbeState::UpToDate(value)
            }
        }
    }

    pub(super) fn durability(&self, db: &<Q as QueryDb<'_>>::DynDb) -> Durability {
        match &*self.state.read() {
            QueryState::NotComputed => Durability::LOW,
            QueryState::InProgress { .. } => panic!("query in progress"),
            QueryState::Memoized(memo) => {
                if memo.check_durability(db.salsa_runtime()) {
                    memo.revisions.durability
                } else {
                    Durability::LOW
                }
            }
        }
    }

    pub(super) fn as_table_entry(&self, key: &Q::Key) -> Option<TableEntry<Q::Key, Q::Value>> {
        match &*self.state.read() {
            QueryState::NotComputed => None,
            QueryState::InProgress { .. } => Some(TableEntry::new(key.clone(), None)),
            QueryState::Memoized(memo) => {
                Some(TableEntry::new(key.clone(), Some(memo.value.clone())))
            }
        }
    }

    pub(super) fn invalidate(&self, new_revision: Revision) -> Option<Durability> {
        tracing::debug!("Slot::invalidate(new_revision = {:?})", new_revision);
        match &mut *self.state.write() {
            QueryState::Memoized(memo) => {
                memo.revisions.untracked = true;
                memo.revisions.inputs = None;
                memo.revisions.changed_at = new_revision;
                Some(memo.revisions.durability)
            }
            QueryState::NotComputed => None,
            QueryState::InProgress { .. } => unreachable!(),
        }
    }

    pub(super) fn maybe_changed_after(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        revision: Revision,
        key: &Q::Key,
    ) -> bool {
        let runtime = db.salsa_runtime();
        let revision_now = runtime.current_revision();

        db.unwind_if_cancelled();

        debug!(
            "maybe_changed_after({:?}) called with revision={:?}, revision_now={:?}",
            self, revision, revision_now,
        );

        // Do an initial probe with just the read-lock.
        //
        // If we find that a cache entry for the value is present
        // but hasn't been verified in this revision, we'll have to
        // do more.
        loop {
            match self.maybe_changed_after_probe(db, self.state.read(), runtime, revision_now) {
                MaybeChangedSinceProbeState::Retry => continue,
                MaybeChangedSinceProbeState::ChangedAt(changed_at) => return changed_at > revision,
                MaybeChangedSinceProbeState::Stale(state) => {
                    drop(state);
                    return self.maybe_changed_after_upgrade(db, revision, key);
                }
            }
        }
    }

    fn maybe_changed_after_probe<StateGuard>(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        state: StateGuard,
        runtime: &Runtime,
        revision_now: Revision,
    ) -> MaybeChangedSinceProbeState<StateGuard>
    where
        StateGuard: Deref<Target = QueryState<Q>>,
    {
        match self.probe(db, state, runtime, revision_now) {
            ProbeState::Retry => MaybeChangedSinceProbeState::Retry,

            ProbeState::Stale(state) => MaybeChangedSinceProbeState::Stale(state),

            // If we know when value last changed, we can return right away.
            // Note that we don't need the actual value to be available.
            ProbeState::UpToDate(StampedValue { value: _, durability: _, changed_at }) => {
                MaybeChangedSinceProbeState::ChangedAt(changed_at)
            }

            // If we have nothing cached, then value may have changed.
            ProbeState::NotComputed(_) => MaybeChangedSinceProbeState::ChangedAt(revision_now),
        }
    }

    fn maybe_changed_after_upgrade(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        revision: Revision,
        key: &Q::Key,
    ) -> bool {
        let runtime = db.salsa_runtime();
        let revision_now = runtime.current_revision();

        // Get an upgradable read lock, which permits other reads but no writers.
        // Probe again. If the value is stale (needs to be verified), then upgrade
        // to a write lock and swap it with InProgress while we work.
        let mut old_memo = match self.maybe_changed_after_probe(
            db,
            self.state.upgradable_read(),
            runtime,
            revision_now,
        ) {
            MaybeChangedSinceProbeState::ChangedAt(changed_at) => return changed_at > revision,

            // If another thread was active, then the cache line is going to be
            // either verified or cleared out. Just recurse to figure out which.
            // Note that we don't need an upgradable read.
            MaybeChangedSinceProbeState::Retry => {
                return self.maybe_changed_after(db, revision, key)
            }

            MaybeChangedSinceProbeState::Stale(state) => {
                type RwLockUpgradableReadGuard<'a, T> =
                    lock_api::RwLockUpgradableReadGuard<'a, RawRwLock, T>;

                let mut state = RwLockUpgradableReadGuard::upgrade(state);
                match std::mem::replace(&mut *state, QueryState::in_progress(runtime.id())) {
                    QueryState::Memoized(old_memo) => old_memo,
                    QueryState::NotComputed | QueryState::InProgress { .. } => unreachable!(),
                }
            }
        };

        let panic_guard = PanicGuard::new(self, runtime);
        let active_query = runtime.push_query(self.database_key_index());

        if old_memo.verify_revisions(db.ops_database(), revision_now, &active_query) {
            let maybe_changed = old_memo.revisions.changed_at > revision;
            panic_guard.proceed(Some(old_memo));
            maybe_changed
        } else {
            // We found that this memoized value may have changed
            // but we have an old value. We can re-run the code and
            // actually *check* if it has changed.
            let StampedValue { changed_at, .. } = self.execute(
                db,
                runtime,
                revision_now,
                active_query,
                panic_guard,
                Some(old_memo),
                key,
            );
            changed_at > revision
        }
    }

    /// Helper: see [`Runtime::try_block_on_or_unwind`].
    fn block_on_or_unwind<MutexGuard>(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        runtime: &Runtime,
        other_id: RuntimeId,
        mutex_guard: MutexGuard,
    ) {
        runtime.block_on_or_unwind(
            db.ops_database(),
            self.database_key_index(),
            other_id,
            mutex_guard,
        )
    }
}

impl<Q> QueryState<Q>
where
    Q: QueryFunction,
{
    fn in_progress(id: RuntimeId) -> Self {
        QueryState::InProgress { id, anyone_waiting: Default::default() }
    }
}

struct PanicGuard<'me, Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    slot: &'me Slot<Q>,
    runtime: &'me Runtime,
}

impl<'me, Q> PanicGuard<'me, Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    fn new(slot: &'me Slot<Q>, runtime: &'me Runtime) -> Self {
        Self { slot, runtime }
    }

    /// Indicates that we have concluded normally (without panicking).
    /// If `opt_memo` is some, then this memo is installed as the new
    /// memoized value. If `opt_memo` is `None`, then the slot is cleared
    /// and has no value.
    fn proceed(mut self, opt_memo: Option<Memo<Q::Value>>) {
        self.overwrite_placeholder(WaitResult::Completed, opt_memo);
        std::mem::forget(self)
    }

    /// Overwrites the `InProgress` placeholder for `key` that we
    /// inserted; if others were blocked, waiting for us to finish,
    /// then notify them.
    fn overwrite_placeholder(&mut self, wait_result: WaitResult, opt_memo: Option<Memo<Q::Value>>) {
        let old_value = {
            let mut write = self.slot.state.write();
            match opt_memo {
                // Replace the `InProgress` marker that we installed with the new
                // memo, thus releasing our unique access to this key.
                Some(memo) => std::mem::replace(&mut *write, QueryState::Memoized(memo)),

                // We had installed an `InProgress` marker, but we panicked before
                // it could be removed. At this point, we therefore "own" unique
                // access to our slot, so we can just remove the key.
                None => std::mem::replace(&mut *write, QueryState::NotComputed),
            }
        };

        match old_value {
            QueryState::InProgress { id, anyone_waiting } => {
                assert_eq!(id, self.runtime.id());

                // NB: As noted on the `store`, `Ordering::Relaxed` is
                // sufficient here. This boolean signals us on whether to
                // acquire a mutex; the mutex will guarantee that all writes
                // we are interested in are visible.
                if anyone_waiting.load(Ordering::Relaxed) {
                    self.runtime
                        .unblock_queries_blocked_on(self.slot.database_key_index(), wait_result);
                }
            }
            _ => panic!(
                "\
Unexpected panic during query evaluation, aborting the process.

Please report this bug to https://github.com/salsa-rs/salsa/issues."
            ),
        }
    }
}

impl<Q> Drop for PanicGuard<'_, Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    fn drop(&mut self) {
        if std::thread::panicking() {
            // We panicked before we could proceed and need to remove `key`.
            self.overwrite_placeholder(WaitResult::Panicked, None)
        } else {
            // If no panic occurred, then panic guard ought to be
            // "forgotten" and so this Drop code should never run.
            panic!(".forget() was not called")
        }
    }
}

impl<V> Memo<V>
where
    V: Clone,
{
    /// Determines whether the value stored in this memo (if any) is still
    /// valid in the current revision. If so, returns a stamped value.
    ///
    /// If needed, this will walk each dependency and
    /// recursively invoke `maybe_changed_after`, which may in turn
    /// re-execute the dependency. This can cause cycles to occur,
    /// so the current query must be pushed onto the
    /// stack to permit cycle detection and recovery: therefore,
    /// takes the `active_query` argument as evidence.
    fn verify_value(
        &mut self,
        db: &dyn Database,
        revision_now: Revision,
        active_query: &ActiveQueryGuard<'_>,
    ) -> Option<StampedValue<V>> {
        if self.verify_revisions(db, revision_now, active_query) {
            Some(StampedValue {
                durability: self.revisions.durability,
                changed_at: self.revisions.changed_at,
                value: self.value.clone(),
            })
        } else {
            None
        }
    }

    /// Determines whether the value represented by this memo is still
    /// valid in the current revision; note that the value itself is
    /// not needed for this check. If needed, this will walk each
    /// dependency and recursively invoke `maybe_changed_after`, which
    /// may in turn re-execute the dependency. This can cause cycles to occur,
    /// so the current query must be pushed onto the
    /// stack to permit cycle detection and recovery: therefore,
    /// takes the `active_query` argument as evidence.
    fn verify_revisions(
        &mut self,
        db: &dyn Database,
        revision_now: Revision,
        _active_query: &ActiveQueryGuard<'_>,
    ) -> bool {
        assert!(self.verified_at != revision_now);
        let verified_at = self.verified_at;

        debug!(
            "verify_revisions: verified_at={:?}, revision_now={:?}, inputs={:#?}",
            verified_at, revision_now, self.revisions.inputs
        );

        if self.check_durability(db.salsa_runtime()) {
            return self.mark_value_as_verified(revision_now);
        }

        match &self.revisions.inputs {
            // We can't validate values that had untracked inputs; just have to
            // re-execute.
            None if self.revisions.untracked => return false,
            None => {}

            // Check whether any of our inputs changed since the
            // **last point where we were verified** (not since we
            // last changed). This is important: if we have
            // memoized values, then an input may have changed in
            // revision R2, but we found that *our* value was the
            // same regardless, so our change date is still
            // R1. But our *verification* date will be R2, and we
            // are only interested in finding out whether the
            // input changed *again*.
            Some(inputs) => {
                let changed_input =
                    inputs.slice.iter().find(|&&input| db.maybe_changed_after(input, verified_at));
                if let Some(input) = changed_input {
                    debug!("validate_memoized_value: `{:?}` may have changed", input);

                    return false;
                }
            }
        };

        self.mark_value_as_verified(revision_now)
    }

    /// True if this memo is known not to have changed based on its durability.
    fn check_durability(&self, runtime: &Runtime) -> bool {
        let last_changed = runtime.last_changed_revision(self.revisions.durability);
        debug!(
            "check_durability(last_changed={:?} <= verified_at={:?}) = {:?}",
            last_changed,
            self.verified_at,
            last_changed <= self.verified_at,
        );
        last_changed <= self.verified_at
    }

    fn mark_value_as_verified(&mut self, revision_now: Revision) -> bool {
        self.verified_at = revision_now;
        true
    }
}

impl<Q> std::fmt::Debug for Slot<Q>
where
    Q: QueryFunction,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "{:?}", Q::default())
    }
}

/// Check that `Slot<Q, >: Send + Sync` as long as
/// `DB::DatabaseData: Send + Sync`, which in turn implies that
/// `Q::Key: Send + Sync`, `Q::Value: Send + Sync`.
#[allow(dead_code)]
fn check_send_sync<Q>()
where
    Q: QueryFunction,

    Q::Key: Send + Sync,
    Q::Value: Send + Sync,
{
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<Slot<Q>>();
}

/// Check that `Slot<Q, >: 'static` as long as
/// `DB::DatabaseData: 'static`, which in turn implies that
/// `Q::Key: 'static`, `Q::Value: 'static`.
#[allow(dead_code)]
fn check_static<Q>()
where
    Q: QueryFunction + 'static,
    Q::Key: 'static,
    Q::Value: 'static,
{
    fn is_static<T: 'static>() {}
    is_static::<Slot<Q>>();
}
