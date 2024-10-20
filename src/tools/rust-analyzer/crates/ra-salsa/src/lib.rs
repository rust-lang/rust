#![allow(clippy::type_complexity)]
#![allow(clippy::question_mark)]
#![allow(missing_docs)]
#![warn(rust_2018_idioms)]

//! The salsa crate is a crate for incremental recomputation.  It
//! permits you to define a "database" of queries with both inputs and
//! values derived from those inputs; as you set the inputs, you can
//! re-execute the derived queries and it will try to re-use results
//! from previous invocations as appropriate.

mod derived;
mod derived_lru;
mod durability;
mod hash;
mod input;
mod intern_id;
mod interned;
mod lru;
mod revision;
mod runtime;
mod storage;

pub mod debug;
/// Items in this module are public for implementation reasons,
/// and are exempt from the SemVer guarantees.
#[doc(hidden)]
pub mod plumbing;

use crate::plumbing::CycleRecoveryStrategy;
use crate::plumbing::DerivedQueryStorageOps;
use crate::plumbing::InputQueryStorageOps;
use crate::plumbing::LruQueryStorageOps;
use crate::plumbing::QueryStorageMassOps;
use crate::plumbing::QueryStorageOps;
pub use crate::revision::Revision;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::panic::AssertUnwindSafe;
use std::panic::{self, UnwindSafe};

pub use crate::durability::Durability;
pub use crate::intern_id::InternId;
pub use crate::interned::{InternKey, InternValue, InternValueTrivial};
pub use crate::runtime::Runtime;
pub use crate::runtime::RuntimeId;
pub use crate::storage::Storage;

/// The base trait which your "query context" must implement. Gives
/// access to the salsa runtime, which you must embed into your query
/// context (along with whatever other state you may require).
pub trait Database: plumbing::DatabaseOps {
    /// This function is invoked at key points in the salsa
    /// runtime. It permits the database to be customized and to
    /// inject logging or other custom behavior.
    fn salsa_event(&self, event_fn: Event) {
        _ = event_fn;
    }

    /// Starts unwinding the stack if the current revision is cancelled.
    ///
    /// This method can be called by query implementations that perform
    /// potentially expensive computations, in order to speed up propagation of
    /// cancellation.
    ///
    /// Cancellation will automatically be triggered by salsa on any query
    /// invocation.
    ///
    /// This method should not be overridden by `Database` implementors. A
    /// `salsa_event` is emitted when this method is called, so that should be
    /// used instead.
    #[inline]
    fn unwind_if_cancelled(&self) {
        let runtime = self.salsa_runtime();
        self.salsa_event(Event {
            runtime_id: runtime.id(),
            kind: EventKind::WillCheckCancellation,
        });

        let current_revision = runtime.current_revision();
        let pending_revision = runtime.pending_revision();
        tracing::debug!(
            "unwind_if_cancelled: current_revision={:?}, pending_revision={:?}",
            current_revision,
            pending_revision
        );
        if pending_revision > current_revision {
            runtime.unwind_cancelled();
        }
    }

    /// Gives access to the underlying salsa runtime.
    ///
    /// This method should not be overridden by `Database` implementors.
    fn salsa_runtime(&self) -> &Runtime {
        self.ops_salsa_runtime()
    }

    /// A "synthetic write" causes the system to act *as though* some
    /// input of durability `durability` has changed. This is mostly
    /// useful for profiling scenarios.
    ///
    /// **WARNING:** Just like an ordinary write, this method triggers
    /// cancellation. If you invoke it while a snapshot exists, it
    /// will block until that snapshot is dropped -- if that snapshot
    /// is owned by the current thread, this could trigger deadlock.
    fn synthetic_write(&mut self, durability: Durability) {
        plumbing::DatabaseOps::synthetic_write(self, durability)
    }
}

/// The `Event` struct identifies various notable things that can
/// occur during salsa execution. Instances of this struct are given
/// to `salsa_event`.
pub struct Event {
    /// The id of the snapshot that triggered the event.  Usually
    /// 1-to-1 with a thread, as well.
    pub runtime_id: RuntimeId,

    /// What sort of event was it.
    pub kind: EventKind,
}

impl Event {
    /// Returns a type that gives a user-readable debug output.
    /// Use like `println!("{:?}", index.debug(db))`.
    pub fn debug<'me, D>(&'me self, db: &'me D) -> impl std::fmt::Debug + 'me
    where
        D: ?Sized + plumbing::DatabaseOps,
    {
        EventDebug { event: self, db }
    }
}

impl fmt::Debug for Event {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Event")
            .field("runtime_id", &self.runtime_id)
            .field("kind", &self.kind)
            .finish()
    }
}

struct EventDebug<'me, D: ?Sized>
where
    D: plumbing::DatabaseOps,
{
    event: &'me Event,
    db: &'me D,
}

impl<'me, D: ?Sized> fmt::Debug for EventDebug<'me, D>
where
    D: plumbing::DatabaseOps,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Event")
            .field("runtime_id", &self.event.runtime_id)
            .field("kind", &self.event.kind.debug(self.db))
            .finish()
    }
}

/// An enum identifying the various kinds of events that can occur.
pub enum EventKind {
    /// Occurs when we found that all inputs to a memoized value are
    /// up-to-date and hence the value can be re-used without
    /// executing the closure.
    ///
    /// Executes before the "re-used" value is returned.
    DidValidateMemoizedValue {
        /// The database-key for the affected value. Implements `Debug`.
        database_key: DatabaseKeyIndex,
    },

    /// Indicates that another thread (with id `other_runtime_id`) is processing the
    /// given query (`database_key`), so we will block until they
    /// finish.
    ///
    /// Executes after we have registered with the other thread but
    /// before they have answered us.
    ///
    /// (NB: you can find the `id` of the current thread via the
    /// `salsa_runtime`)
    WillBlockOn {
        /// The id of the runtime we will block on.
        other_runtime_id: RuntimeId,

        /// The database-key for the affected value. Implements `Debug`.
        database_key: DatabaseKeyIndex,
    },

    /// Indicates that the function for this query will be executed.
    /// This is either because it has never executed before or because
    /// its inputs may be out of date.
    WillExecute {
        /// The database-key for the affected value. Implements `Debug`.
        database_key: DatabaseKeyIndex,
    },

    /// Indicates that `unwind_if_cancelled` was called and salsa will check if
    /// the current revision has been cancelled.
    WillCheckCancellation,
}

impl EventKind {
    /// Returns a type that gives a user-readable debug output.
    /// Use like `println!("{:?}", index.debug(db))`.
    pub fn debug<'me, D>(&'me self, db: &'me D) -> impl std::fmt::Debug + 'me
    where
        D: ?Sized + plumbing::DatabaseOps,
    {
        EventKindDebug { kind: self, db }
    }
}

impl fmt::Debug for EventKind {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventKind::DidValidateMemoizedValue { database_key } => fmt
                .debug_struct("DidValidateMemoizedValue")
                .field("database_key", database_key)
                .finish(),
            EventKind::WillBlockOn { other_runtime_id, database_key } => fmt
                .debug_struct("WillBlockOn")
                .field("other_runtime_id", other_runtime_id)
                .field("database_key", database_key)
                .finish(),
            EventKind::WillExecute { database_key } => {
                fmt.debug_struct("WillExecute").field("database_key", database_key).finish()
            }
            EventKind::WillCheckCancellation => fmt.debug_struct("WillCheckCancellation").finish(),
        }
    }
}

struct EventKindDebug<'me, D: ?Sized>
where
    D: plumbing::DatabaseOps,
{
    kind: &'me EventKind,
    db: &'me D,
}

impl<'me, D: ?Sized> fmt::Debug for EventKindDebug<'me, D>
where
    D: plumbing::DatabaseOps,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            EventKind::DidValidateMemoizedValue { database_key } => fmt
                .debug_struct("DidValidateMemoizedValue")
                .field("database_key", &database_key.debug(self.db))
                .finish(),
            EventKind::WillBlockOn { other_runtime_id, database_key } => fmt
                .debug_struct("WillBlockOn")
                .field("other_runtime_id", &other_runtime_id)
                .field("database_key", &database_key.debug(self.db))
                .finish(),
            EventKind::WillExecute { database_key } => fmt
                .debug_struct("WillExecute")
                .field("database_key", &database_key.debug(self.db))
                .finish(),
            EventKind::WillCheckCancellation => fmt.debug_struct("WillCheckCancellation").finish(),
        }
    }
}

/// Indicates a database that also supports parallel query
/// evaluation. All of Salsa's base query support is capable of
/// parallel execution, but for it to work, your query key/value types
/// must also be `Send`, as must any additional data in your database.
pub trait ParallelDatabase: Database + Send {
    /// Creates a second handle to the database that holds the
    /// database fixed at a particular revision. So long as this
    /// "frozen" handle exists, any attempt to [`set`] an input will
    /// block.
    ///
    /// [`set`]: struct.QueryTable.html#method.set
    ///
    /// This is the method you are meant to use most of the time in a
    /// parallel setting where modifications may arise asynchronously
    /// (e.g., a language server). In this context, it is common to
    /// wish to "fork off" a snapshot of the database performing some
    /// series of queries in parallel and arranging the results. Using
    /// this method for that purpose ensures that those queries will
    /// see a consistent view of the database (it is also advisable
    /// for those queries to use the [`Database::unwind_if_cancelled`]
    /// method to check for cancellation).
    ///
    /// # Panics
    ///
    /// It is not permitted to create a snapshot from inside of a
    /// query. Attepting to do so will panic.
    ///
    /// # Deadlock warning
    ///
    /// The intended pattern for snapshots is that, once created, they
    /// are sent to another thread and used from there. As such, the
    /// `snapshot` acquires a "read lock" on the database --
    /// therefore, so long as the `snapshot` is not dropped, any
    /// attempt to `set` a value in the database will block. If the
    /// `snapshot` is owned by the same thread that is attempting to
    /// `set`, this will cause a problem.
    ///
    /// # How to implement this
    ///
    /// Typically, this method will create a second copy of your
    /// database type (`MyDatabaseType`, in the example below),
    /// cloning over each of the fields from `self` into this new
    /// copy. For the field that stores the salsa runtime, you should
    /// use [the `Runtime::snapshot` method][rfm] to create a snapshot of the
    /// runtime. Finally, package up the result using `Snapshot::new`,
    /// which is a simple wrapper type that only gives `&self` access
    /// to the database within (thus preventing the use of methods
    /// that may mutate the inputs):
    ///
    /// [rfm]: struct.Runtime.html#method.snapshot
    ///
    /// ```rust,ignore
    /// impl ParallelDatabase for MyDatabaseType {
    ///     fn snapshot(&self) -> Snapshot<Self> {
    ///         Snapshot::new(
    ///             MyDatabaseType {
    ///                 runtime: self.runtime.snapshot(self),
    ///                 other_field: self.other_field.clone(),
    ///             }
    ///         )
    ///     }
    /// }
    /// ```
    fn snapshot(&self) -> Snapshot<Self>;
}

/// Simple wrapper struct that takes ownership of a database `DB` and
/// only gives `&self` access to it. See [the `snapshot` method][fm]
/// for more details.
///
/// [fm]: trait.ParallelDatabase.html#method.snapshot
#[derive(Debug)]
pub struct Snapshot<DB: ?Sized>
where
    DB: ParallelDatabase,
{
    db: DB,
}

impl<DB> Snapshot<DB>
where
    DB: ParallelDatabase,
{
    /// Creates a `Snapshot` that wraps the given database handle
    /// `db`. From this point forward, only shared references to `db`
    /// will be possible.
    pub fn new(db: DB) -> Self {
        Snapshot { db }
    }
}

impl<DB> std::ops::Deref for Snapshot<DB>
where
    DB: ParallelDatabase,
{
    type Target = DB;

    fn deref(&self) -> &DB {
        &self.db
    }
}

/// An integer that uniquely identifies a particular query instance within the
/// database. Used to track dependencies between queries. Fully ordered and
/// equatable but those orderings are arbitrary, and meant to be used only for
/// inserting into maps and the like.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DatabaseKeyIndex {
    group_index: u16,
    query_index: u16,
    key_index: u32,
}

impl DatabaseKeyIndex {
    /// Returns the index of the query group containing this key.
    #[inline]
    pub fn group_index(self) -> u16 {
        self.group_index
    }

    /// Returns the index of the query within its query group.
    #[inline]
    pub fn query_index(self) -> u16 {
        self.query_index
    }

    /// Returns the index of this particular query key within the query.
    #[inline]
    pub fn key_index(self) -> u32 {
        self.key_index
    }

    /// Returns a type that gives a user-readable debug output.
    /// Use like `println!("{:?}", index.debug(db))`.
    pub fn debug<D>(self, db: &D) -> impl std::fmt::Debug + '_
    where
        D: ?Sized + plumbing::DatabaseOps,
    {
        DatabaseKeyIndexDebug { index: self, db }
    }
}

/// Helper type for `DatabaseKeyIndex::debug`
struct DatabaseKeyIndexDebug<'me, D: ?Sized>
where
    D: plumbing::DatabaseOps,
{
    index: DatabaseKeyIndex,
    db: &'me D,
}

impl<D: ?Sized> std::fmt::Debug for DatabaseKeyIndexDebug<'_, D>
where
    D: plumbing::DatabaseOps,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.db.fmt_index(self.index, fmt)
    }
}

/// Trait implements by all of the "special types" associated with
/// each of your queries.
///
/// Base trait of `Query` that has a lifetime parameter to allow the `DynDb` to be non-'static.
pub trait QueryDb<'d>: Sized {
    /// Dyn version of the associated trait for this query group.
    type DynDb: ?Sized + Database + HasQueryGroup<Self::Group> + 'd;

    /// Associate query group struct.
    type Group: plumbing::QueryGroup<GroupStorage = Self::GroupStorage>;

    /// Generated struct that contains storage for all queries in a group.
    type GroupStorage;
}

/// Trait implements by all of the "special types" associated with
/// each of your queries.
pub trait Query: Debug + Default + Sized + for<'d> QueryDb<'d> {
    /// Type that you give as a parameter -- for queries with zero
    /// or more than one input, this will be a tuple.
    type Key: Clone + Debug + Hash + Eq;

    /// What value does the query return?
    type Value: Clone + Debug;

    /// Internal struct storing the values for the query.
    // type Storage: plumbing::QueryStorageOps<Self>;
    type Storage;

    /// A unique index identifying this query within the group.
    const QUERY_INDEX: u16;

    /// Name of the query method (e.g., `foo`)
    const QUERY_NAME: &'static str;

    /// Extract storage for this query from the storage for its group.
    fn query_storage<'a>(
        group_storage: &'a <Self as QueryDb<'_>>::GroupStorage,
    ) -> &'a std::sync::Arc<Self::Storage>;

    /// Extract storage for this query from the storage for its group.
    fn query_storage_mut<'a>(
        group_storage: &'a <Self as QueryDb<'_>>::GroupStorage,
    ) -> &'a std::sync::Arc<Self::Storage>;
}

/// Return value from [the `query` method] on `Database`.
/// Gives access to various less common operations on queries.
///
/// [the `query` method]: trait.Database.html#method.query
pub struct QueryTable<'me, Q>
where
    Q: Query,
{
    db: &'me <Q as QueryDb<'me>>::DynDb,
    storage: &'me Q::Storage,
}

impl<'me, Q> QueryTable<'me, Q>
where
    Q: Query,
    Q::Storage: QueryStorageOps<Q>,
{
    /// Constructs a new `QueryTable`.
    pub fn new(db: &'me <Q as QueryDb<'me>>::DynDb, storage: &'me Q::Storage) -> Self {
        Self { db, storage }
    }

    /// Execute the query on a given input. Usually it's easier to
    /// invoke the trait method directly. Note that for variadic
    /// queries (those with no inputs, or those with more than one
    /// input) the key will be a tuple.
    pub fn get(&self, key: Q::Key) -> Q::Value {
        self.storage.fetch(self.db, &key)
    }

    /// Completely clears the storage for this query.
    ///
    /// This method breaks internal invariants of salsa, so any further queries
    /// might return nonsense results. It is useful only in very specific
    /// circumstances -- for example, when one wants to observe which values
    /// dropped together with the table
    pub fn purge(&self)
    where
        Q::Storage: plumbing::QueryStorageMassOps,
    {
        self.storage.purge();
    }

    pub fn storage(&self) -> &<Q as Query>::Storage {
        self.storage
    }
}

/// Return value from [the `query_mut` method] on `Database`.
/// Gives access to the `set` method, notably, that is used to
/// set the value of an input query.
///
/// [the `query_mut` method]: trait.Database.html#method.query_mut
pub struct QueryTableMut<'me, Q>
where
    Q: Query + 'me,
{
    runtime: &'me mut Runtime,
    storage: &'me Q::Storage,
}

impl<'me, Q> QueryTableMut<'me, Q>
where
    Q: Query,
{
    /// Constructs a new `QueryTableMut`.
    pub fn new(runtime: &'me mut Runtime, storage: &'me Q::Storage) -> Self {
        Self { runtime, storage }
    }

    /// Assign a value to an "input query". Must be used outside of
    /// an active query computation.
    ///
    /// If you are using `snapshot`, see the notes on blocking
    /// and cancellation on [the `query_mut` method].
    ///
    /// [the `query_mut` method]: trait.Database.html#method.query_mut
    pub fn set(&mut self, key: Q::Key, value: Q::Value)
    where
        Q::Storage: plumbing::InputQueryStorageOps<Q>,
    {
        self.set_with_durability(key, value, Durability::LOW);
    }

    /// Assign a value to an "input query", with the additional
    /// promise that this value will **never change**. Must be used
    /// outside of an active query computation.
    ///
    /// If you are using `snapshot`, see the notes on blocking
    /// and cancellation on [the `query_mut` method].
    ///
    /// [the `query_mut` method]: trait.Database.html#method.query_mut
    pub fn set_with_durability(&mut self, key: Q::Key, value: Q::Value, durability: Durability)
    where
        Q::Storage: plumbing::InputQueryStorageOps<Q>,
    {
        self.storage.set(self.runtime, &key, value, durability);
    }

    /// Sets the size of LRU cache of values for this query table.
    ///
    /// That is, at most `cap` values will be preset in the table at the same
    /// time. This helps with keeping maximum memory usage under control, at the
    /// cost of potential extra recalculations of evicted values.
    ///
    /// If `cap` is zero, all values are preserved, this is the default.
    pub fn set_lru_capacity(&self, cap: u16)
    where
        Q::Storage: plumbing::LruQueryStorageOps,
    {
        self.storage.set_lru_capacity(cap);
    }

    /// Marks the computed value as outdated.
    ///
    /// This causes salsa to re-execute the query function on the next access to
    /// the query, even if all dependencies are up to date.
    ///
    /// This is most commonly used as part of the [on-demand input
    /// pattern](https://salsa-rs.github.io/salsa/common_patterns/on_demand_inputs.html).
    pub fn invalidate(&mut self, key: &Q::Key)
    where
        Q::Storage: plumbing::DerivedQueryStorageOps<Q>,
    {
        self.storage.invalidate(self.runtime, key)
    }
}

/// A panic payload indicating that execution of a salsa query was cancelled.
///
/// This can occur for a few reasons:
/// *
/// *
/// *
#[derive(Debug)]
#[non_exhaustive]
pub enum Cancelled {
    /// The query was operating on revision R, but there is a pending write to move to revision R+1.
    #[non_exhaustive]
    PendingWrite,

    /// The query was blocked on another thread, and that thread panicked.
    #[non_exhaustive]
    PropagatedPanic,
}

impl Cancelled {
    fn throw(self) -> ! {
        // We use resume and not panic here to avoid running the panic
        // hook (that is, to avoid collecting and printing backtrace).
        std::panic::resume_unwind(Box::new(self));
    }

    /// Runs `f`, and catches any salsa cancellation.
    pub fn catch<F, T>(f: F) -> Result<T, Cancelled>
    where
        F: FnOnce() -> T + UnwindSafe,
    {
        match panic::catch_unwind(f) {
            Ok(t) => Ok(t),
            Err(payload) => match payload.downcast() {
                Ok(cancelled) => Err(*cancelled),
                Err(payload) => panic::resume_unwind(payload),
            },
        }
    }
}

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let why = match self {
            Cancelled::PendingWrite => "pending write",
            Cancelled::PropagatedPanic => "propagated panic",
        };
        f.write_str("cancelled because of ")?;
        f.write_str(why)
    }
}

impl std::error::Error for Cancelled {}

/// Captures the participants of a cycle that occurred when executing a query.
///
/// This type is meant to be used to help give meaningful error messages to the
/// user or to help salsa developers figure out why their program is resulting
/// in a computation cycle.
///
/// It is used in a few ways:
///
/// * During [cycle recovery](https://https://salsa-rs.github.io/salsa/cycles/fallback.html),
///   where it is given to the fallback function.
/// * As the panic value when an unexpected cycle (i.e., a cycle where one or more participants
///   lacks cycle recovery information) occurs.
///
/// You can read more about cycle handling in
/// the [salsa book](https://https://salsa-rs.github.io/salsa/cycles.html).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cycle {
    participants: plumbing::CycleParticipants,
}

impl Cycle {
    pub(crate) fn new(participants: plumbing::CycleParticipants) -> Self {
        Self { participants }
    }

    /// True if two `Cycle` values represent the same cycle.
    pub(crate) fn is(&self, cycle: &Cycle) -> bool {
        triomphe::Arc::ptr_eq(&self.participants, &cycle.participants)
    }

    pub(crate) fn throw(self) -> ! {
        tracing::debug!("throwing cycle {:?}", self);
        std::panic::resume_unwind(Box::new(self))
    }

    pub(crate) fn catch<T>(execute: impl FnOnce() -> T) -> Result<T, Cycle> {
        match std::panic::catch_unwind(AssertUnwindSafe(execute)) {
            Ok(v) => Ok(v),
            Err(err) => match err.downcast::<Cycle>() {
                Ok(cycle) => Err(*cycle),
                Err(other) => std::panic::resume_unwind(other),
            },
        }
    }

    /// Iterate over the [`DatabaseKeyIndex`] for each query participating
    /// in the cycle. The start point of this iteration within the cycle
    /// is arbitrary but deterministic, but the ordering is otherwise determined
    /// by the execution.
    pub fn participant_keys(&self) -> impl Iterator<Item = DatabaseKeyIndex> + '_ {
        self.participants.iter().copied()
    }

    /// Returns a vector with the debug information for
    /// all the participants in the cycle.
    pub fn all_participants<DB: ?Sized + Database>(&self, db: &DB) -> Vec<String> {
        self.participant_keys().map(|d| format!("{:?}", d.debug(db))).collect()
    }

    /// Returns a vector with the debug information for
    /// those participants in the cycle that lacked recovery
    /// information.
    pub fn unexpected_participants<DB: ?Sized + Database>(&self, db: &DB) -> Vec<String> {
        self.participant_keys()
            .filter(|&d| db.cycle_recovery_strategy(d) == CycleRecoveryStrategy::Panic)
            .map(|d| format!("{:?}", d.debug(db)))
            .collect()
    }

    /// Returns a "debug" view onto this strict that can be used to print out information.
    pub fn debug<'me, DB: ?Sized + Database>(&'me self, db: &'me DB) -> impl std::fmt::Debug + 'me {
        struct UnexpectedCycleDebug<'me> {
            c: &'me Cycle,
            db: &'me dyn Database,
        }

        impl<'me> std::fmt::Debug for UnexpectedCycleDebug<'me> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                fmt.debug_struct("UnexpectedCycle")
                    .field("all_participants", &self.c.all_participants(self.db))
                    .field("unexpected_participants", &self.c.unexpected_participants(self.db))
                    .finish()
            }
        }

        UnexpectedCycleDebug { c: self, db: db.ops_database() }
    }
}

// Re-export the procedural macros.
#[allow(unused_imports)]
#[macro_use]
extern crate ra_salsa_macros;
use plumbing::HasQueryGroup;
pub use ra_salsa_macros::*;
