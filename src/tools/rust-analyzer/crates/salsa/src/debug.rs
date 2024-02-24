//! Debugging APIs: these are meant for use when unit-testing or
//! debugging your application but aren't ordinarily needed.

use crate::durability::Durability;
use crate::plumbing::QueryStorageOps;
use crate::Query;
use crate::QueryTable;

/// Additional methods on queries that can be used to "peek into"
/// their current state. These methods are meant for debugging and
/// observing the effects of garbage collection etc.
pub trait DebugQueryTable {
    /// Key of this query.
    type Key;

    /// Value of this query.
    type Value;

    /// Returns a lower bound on the durability for the given key.
    /// This is typically the minimum durability of all values that
    /// the query accessed, but we may return a lower durability in
    /// some cases.
    fn durability(&self, key: Self::Key) -> Durability;

    /// Get the (current) set of the entries in the query table.
    fn entries<C>(&self) -> C
    where
        C: FromIterator<TableEntry<Self::Key, Self::Value>>;
}

/// An entry from a query table, for debugging and inspecting the table state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub struct TableEntry<K, V> {
    /// key of the query
    pub key: K,
    /// value of the query, if it is stored
    pub value: Option<V>,
}

impl<K, V> TableEntry<K, V> {
    pub(crate) fn new(key: K, value: Option<V>) -> TableEntry<K, V> {
        TableEntry { key, value }
    }
}

impl<Q> DebugQueryTable for QueryTable<'_, Q>
where
    Q: Query,
    Q::Storage: QueryStorageOps<Q>,
{
    type Key = Q::Key;
    type Value = Q::Value;

    fn durability(&self, key: Q::Key) -> Durability {
        self.storage.durability(self.db, &key)
    }

    fn entries<C>(&self) -> C
    where
        C: FromIterator<TableEntry<Self::Key, Self::Value>>,
    {
        self.storage.entries(self.db)
    }
}
