use crate::{plumbing::DatabaseStorageTypes, Runtime};
use triomphe::Arc;

/// Stores the cached results and dependency information for all the queries
/// defined on your salsa database. Also embeds a [`Runtime`] which is used to
/// manage query execution. Every database must include a `storage:
/// Storage<Self>` field.
pub struct Storage<DB: DatabaseStorageTypes> {
    query_store: Arc<DB::DatabaseStorage>,
    runtime: Runtime,
}

impl<DB: DatabaseStorageTypes> Default for Storage<DB> {
    fn default() -> Self {
        Self { query_store: Default::default(), runtime: Default::default() }
    }
}

impl<DB: DatabaseStorageTypes> Storage<DB> {
    /// Gives access to the underlying salsa runtime.
    pub fn salsa_runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Gives access to the underlying salsa runtime.
    pub fn salsa_runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// Access the query storage tables. Not meant to be used directly by end
    /// users.
    pub fn query_store(&self) -> &DB::DatabaseStorage {
        &self.query_store
    }

    /// Access the query storage tables. Not meant to be used directly by end
    /// users.
    pub fn query_store_mut(&mut self) -> (&DB::DatabaseStorage, &mut Runtime) {
        (&self.query_store, &mut self.runtime)
    }

    /// Returns a "snapshotted" storage, suitable for use in a forked database.
    /// This snapshot hold a read-lock on the global state, which means that any
    /// attempt to `set` an input will block until the forked runtime is
    /// dropped. See `ParallelDatabase::snapshot` for more information.
    ///
    /// **Warning.** This second handle is intended to be used from a separate
    /// thread. Using two database handles from the **same thread** can lead to
    /// deadlock.
    pub fn snapshot(&self) -> Self {
        Storage { query_store: self.query_store.clone(), runtime: self.runtime.snapshot() }
    }
}
