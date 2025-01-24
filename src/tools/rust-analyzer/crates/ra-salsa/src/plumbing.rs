#![allow(missing_docs)]

use crate::debug::TableEntry;
use crate::durability::Durability;
use crate::Cycle;
use crate::Database;
use crate::Query;
use crate::QueryTable;
use crate::QueryTableMut;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use triomphe::Arc;

pub use crate::derived::MemoizedStorage;
pub use crate::derived_lru::DependencyStorage as LruDependencyStorage;
pub use crate::derived_lru::MemoizedStorage as LruMemoizedStorage;
pub use crate::input::{InputStorage, UnitInputStorage};
pub use crate::interned::InternedStorage;
pub use crate::interned::LookupInternedStorage;
pub use crate::{revision::Revision, DatabaseKeyIndex, QueryDb, Runtime};

/// Defines various associated types. An impl of this
/// should be generated for your query-context type automatically by
/// the `database_storage` macro, so you shouldn't need to mess
/// with this trait directly.
pub trait DatabaseStorageTypes: Database {
    /// Defines the "storage type", where all the query data is kept.
    /// This type is defined by the `database_storage` macro.
    type DatabaseStorage: Default;
}

/// Internal operations that the runtime uses to operate on the database.
pub trait DatabaseOps {
    /// Upcast this type to a `dyn Database`.
    fn ops_database(&self) -> &dyn Database;

    /// Gives access to the underlying salsa runtime.
    fn ops_salsa_runtime(&self) -> &Runtime;

    /// A "synthetic write" causes the system to act *as though* some
    /// input of durability `durability` has changed. This is mostly
    /// useful for profiling scenarios.
    ///
    /// **WARNING:** Just like an ordinary write, this method triggers
    /// cancellation. If you invoke it while a snapshot exists, it
    /// will block until that snapshot is dropped -- if that snapshot
    /// is owned by the current thread, this could trigger deadlock.
    fn synthetic_write(&mut self, durability: Durability);

    /// Formats a database key index in a human readable fashion.
    fn fmt_index(
        &self,
        index: DatabaseKeyIndex,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;

    /// True if the computed value for `input` may have changed since `revision`.
    fn maybe_changed_after(&self, input: DatabaseKeyIndex, revision: Revision) -> bool;

    /// Find the `CycleRecoveryStrategy` for a given input.
    fn cycle_recovery_strategy(&self, input: DatabaseKeyIndex) -> CycleRecoveryStrategy;

    /// Executes the callback for each kind of query.
    fn for_each_query(&self, op: &mut dyn FnMut(&dyn QueryStorageMassOps));
}

/// Internal operations performed on the query storage as a whole
/// (note that these ops do not need to know the identity of the
/// query, unlike `QueryStorageOps`).
pub trait QueryStorageMassOps {
    fn purge(&self);
}

pub trait DatabaseKey: Clone + Debug + Eq + Hash {}

pub trait QueryFunction: Query {
    /// See `CycleRecoveryStrategy`
    const CYCLE_STRATEGY: CycleRecoveryStrategy;

    fn execute(db: &<Self as QueryDb<'_>>::DynDb, key: Self::Key) -> Self::Value;

    fn cycle_fallback(
        db: &<Self as QueryDb<'_>>::DynDb,
        cycle: &Cycle,
        key: &Self::Key,
    ) -> Self::Value {
        let _ = (db, cycle, key);
        panic!("query `{:?}` doesn't support cycle fallback", Self::default())
    }
}

/// Cycle recovery strategy: Is this query capable of recovering from
/// a cycle that results from executing the function? If so, how?
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CycleRecoveryStrategy {
    /// Cannot recover from cycles: panic.
    ///
    /// This is the default. It is also what happens if a cycle
    /// occurs and the queries involved have different recovery
    /// strategies.
    ///
    /// In the case of a failure due to a cycle, the panic
    /// value will be XXX (FIXME).
    Panic,

    /// Recovers from cycles by storing a sentinel value.
    ///
    /// This value is computed by the `QueryFunction::cycle_fallback`
    /// function.
    Fallback,
}

/// Create a query table, which has access to the storage for the query
/// and offers methods like `get`.
pub fn get_query_table<'me, Q>(db: &'me <Q as QueryDb<'me>>::DynDb) -> QueryTable<'me, Q>
where
    Q: Query + 'me,
    Q::Storage: QueryStorageOps<Q>,
{
    let group_storage: &Q::GroupStorage = HasQueryGroup::group_storage(db);
    let query_storage: &Q::Storage = Q::query_storage(group_storage);
    QueryTable::new(db, query_storage)
}

/// Create a mutable query table, which has access to the storage
/// for the query and offers methods like `set`.
pub fn get_query_table_mut<'me, Q>(db: &'me mut <Q as QueryDb<'me>>::DynDb) -> QueryTableMut<'me, Q>
where
    Q: Query,
{
    let (group_storage, runtime) = HasQueryGroup::group_storage_mut(db);
    let query_storage = Q::query_storage_mut(group_storage);
    QueryTableMut::new(runtime, &**query_storage)
}

pub trait QueryGroup: Sized {
    type GroupStorage;

    /// Dyn version of the associated database trait.
    type DynDb: ?Sized + Database + HasQueryGroup<Self>;
}

/// Trait implemented by a database for each group that it supports.
/// `S` and `K` are the types for *group storage* and *group key*, respectively.
pub trait HasQueryGroup<G>: Database
where
    G: QueryGroup,
{
    /// Access the group storage struct from the database.
    fn group_storage(&self) -> &G::GroupStorage;

    /// Access the group storage struct from the database.
    /// Also returns a ref to the `Runtime`, since otherwise
    /// the database is borrowed and one cannot get access to it.
    fn group_storage_mut(&mut self) -> (&G::GroupStorage, &mut Runtime);
}

// ANCHOR:QueryStorageOps
pub trait QueryStorageOps<Q>
where
    Self: QueryStorageMassOps,
    Q: Query,
{
    // ANCHOR_END:QueryStorageOps

    /// See CycleRecoveryStrategy
    const CYCLE_STRATEGY: CycleRecoveryStrategy;

    fn new(group_index: u16) -> Self;

    /// Format a database key index in a suitable way.
    fn fmt_index(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        index: u32,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;

    // ANCHOR:maybe_changed_after
    /// True if the value of `input`, which must be from this query, may have
    /// changed after the given revision ended.
    ///
    /// This function should only be invoked with a revision less than the current
    /// revision.
    fn maybe_changed_after(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        index: u32,
        revision: Revision,
    ) -> bool;
    // ANCHOR_END:maybe_changed_after

    fn cycle_recovery_strategy(&self) -> CycleRecoveryStrategy {
        Self::CYCLE_STRATEGY
    }

    // ANCHOR:fetch
    /// Execute the query, returning the result (often, the result
    /// will be memoized).  This is the "main method" for
    /// queries.
    ///
    /// Returns `Err` in the event of a cycle, meaning that computing
    /// the value for this `key` is recursively attempting to fetch
    /// itself.
    fn fetch(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Q::Value;
    // ANCHOR_END:fetch

    /// Returns the durability associated with a given key.
    fn durability(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Durability;

    /// Get the (current) set of the entries in the query storage
    fn entries<C>(&self, db: &<Q as QueryDb<'_>>::DynDb) -> C
    where
        C: std::iter::FromIterator<TableEntry<Q::Key, Q::Value>>;
}

/// An optional trait that is implemented for "user mutable" storage:
/// that is, storage whose value is not derived from other storage but
/// is set independently.
pub trait InputQueryStorageOps<Q>
where
    Q: Query,
{
    fn set(&self, runtime: &mut Runtime, key: &Q::Key, new_value: Q::Value, durability: Durability);
}

/// An optional trait that is implemented for "user mutable" storage:
/// that is, storage whose value is not derived from other storage but
/// is set independently.
pub trait LruQueryStorageOps {
    fn set_lru_capacity(&self, new_capacity: u16);
}

pub trait DerivedQueryStorageOps<Q>
where
    Q: Query,
{
    fn invalidate<S>(&self, runtime: &mut Runtime, key: &S)
    where
        S: Eq + Hash,
        Q::Key: Borrow<S>;
}

pub type CycleParticipants = Arc<Vec<DatabaseKeyIndex>>;
