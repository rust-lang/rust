use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};

use horde::collect::{Pin, pin};
pub use horde::sync_table::Read;
use horde::sync_table::Write;
use rustc_hash::FxBuildHasher;

use crate::sharded::IntoPointer;
use crate::sync::{DynSync, Lock, LockGuard};

pub struct SyncTable<K, V> {
    // We use this lock to protect `table` instead of the internal mutex in `horde::SyncTable`
    // as it's faster when synchronization is disabled.
    lock: Lock<()>,

    table: horde::SyncTable<K, V, FxBuildHasher>,
}

// Memory reclamation can move elements to other threads for dropping,
// so we require `Sync` instead of `DynSync` here
unsafe impl<K: Sync, V: Sync> DynSync for SyncTable<K, V> where FxBuildHasher: Sync {}

impl<K, V> Default for SyncTable<K, V> {
    fn default() -> Self {
        Self { lock: Lock::default(), table: horde::SyncTable::default() }
    }
}

impl<K, V> SyncTable<K, V> {
    /// Creates a [Read] handle from a pinned region.
    ///
    /// Use [horde::collect::pin] to get a `Pin` instance.
    #[inline]
    pub fn read<'a>(&'a self, pin: Pin<'a>) -> Read<'a, K, V, FxBuildHasher> {
        self.table.read(pin)
    }

    /// Creates a [LockedWrite] handle by taking the underlying mutex that protects writes.
    #[inline]
    pub fn lock(&self) -> LockedWrite<'_, K, V> {
        LockedWrite {
            _guard: self.lock.lock(),
            table: {
                // SAFETY: We ensure there's only 1 writer at a time using our own lock
                unsafe { self.table.unsafe_write() }
            },
        }
    }

    /// Hashes a key with the table's hasher.
    #[inline]
    pub fn hash_key<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash,
    {
        self.table.hash_key::<Q>(key)
    }

    pub fn len(&self) -> usize {
        pin(|pin| self.read(pin).len())
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { lock: Lock::new(()), table: horde::SyncTable::new_with(FxBuildHasher, cap) }
    }
}

/// A handle to a [SyncTable] with write access protected by a lock.
pub struct LockedWrite<'a, K, V> {
    table: Write<'a, K, V, FxBuildHasher>,
    _guard: LockGuard<'a, ()>,
}

impl<'a, K, V> Deref for LockedWrite<'a, K, V> {
    type Target = Write<'a, K, V, FxBuildHasher>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl<'a, K, V> DerefMut for LockedWrite<'a, K, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

impl<K: Eq + Hash + Copy + Send> SyncTable<K, ()> {
    pub fn contains_pointer_to<T: Hash + IntoPointer>(&self, value: &T) -> bool
    where
        K: IntoPointer,
    {
        pin(|pin| {
            let mut state = FxBuildHasher.build_hasher();
            value.hash(&mut state);
            let hash = state.finish();
            let value = value.into_pointer();
            self.read(pin).get_from_hash(hash, |entry| entry.into_pointer() == value).is_some()
        })
    }

    #[inline]
    pub fn intern_ref<Q: ?Sized>(&self, value: &Q, make: impl FnOnce() -> K) -> K
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        pin(|pin| {
            let hash = self.hash_key(value);

            let entry = self.read(pin).get(value, Some(hash));
            if let Some(entry) = entry {
                return *entry.0;
            }

            let mut write = self.lock();

            let entry = self.read(pin).get(value, Some(hash));
            if let Some(entry) = entry {
                return *entry.0;
            }

            let result = make();

            write.insert_new(result, (), Some(hash));

            result
        })
    }

    #[inline]
    pub fn intern<Q>(&self, value: Q, make: impl FnOnce(Q) -> K) -> K
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        pin(|pin| {
            let hash = self.hash_key(&value);

            let entry = self.read(pin).get(&value, Some(hash));
            if let Some(entry) = entry {
                return *entry.0;
            }

            let mut write = self.lock();

            let entry = self.read(pin).get(&value, Some(hash));
            if let Some(entry) = entry {
                return *entry.0;
            }

            let result = make(value);

            write.insert_new(result, (), Some(hash));

            result
        })
    }
}
