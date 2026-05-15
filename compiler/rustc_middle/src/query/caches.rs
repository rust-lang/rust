use rustc_arena::TypedArena;
use rustc_data_structures::cache_entry::CacheEntry;
use rustc_data_structures::sharded::ShardedHashMap;
use rustc_data_structures::sync::{DynSend, DynSync, WorkerLocal};
pub use rustc_data_structures::vec_cache::VecCache;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_index::Idx;
use rustc_span::def_id::{DefId, DefIndex};

use crate::dep_graph::DepNodeIndex;
use crate::query::keys::QueryKey;

/// Trait for types that serve as an in-memory cache for query results,
/// for a given key (argument) type and value (return) type.
///
/// Types implementing this trait are associated with actual key/value types
/// by the `Cache` associated type of the `rustc_middle::query::Key` trait.
pub trait QueryCache: Sized {
    type Key: QueryKey;
    type Value: Copy;

    /// Returns the cached value (and other information) associated with the
    /// given key, if it is present in the cache.
    fn lookup(&self, key: Self::Key) -> &CacheEntry<Self::Value>;

    /// Calls a closure on each entry in this cache.
    fn for_each(&self, f: impl FnMut(Self::Key, &Self::Value, DepNodeIndex));
}

struct SyncConstPtr<T>(*const T);

impl<T> Copy for SyncConstPtr<T> {}

impl<T> Clone for SyncConstPtr<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
unsafe impl<T: Sync> Send for SyncConstPtr<T> {}
unsafe impl<T: Sync> Sync for SyncConstPtr<T> {}
unsafe impl<T: DynSync> DynSend for SyncConstPtr<T> {}
unsafe impl<T: DynSync> DynSync for SyncConstPtr<T> {}

/// In-memory cache for queries whose keys aren't suitable for any of the
/// more specialized kinds of cache. Backed by a sharded hashmap.
pub struct DefaultCache<K, V> {
    cache: ShardedHashMap<K, SyncConstPtr<CacheEntry<V>>>,
    arena: WorkerLocal<TypedArena<CacheEntry<V>>>,
}

impl<K, V> Default for DefaultCache<K, V> {
    fn default() -> Self {
        DefaultCache { cache: Default::default(), arena: Default::default() }
    }
}

impl<K, V: Copy> QueryCache for DefaultCache<K, V>
where
    K: QueryKey,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: K) -> &CacheEntry<V> {
        // FIXME: zero out arena to avoid writes
        unsafe {
            &*self
                .cache
                .get_or_insert_with(key, || SyncConstPtr(self.arena.alloc(CacheEntry::empty())))
                .0
        }
    }

    fn for_each(&self, mut f: impl FnMut(Self::Key, &Self::Value, DepNodeIndex)) {
        for shard in self.cache.lock_shards() {
            for &(k, ent) in shard.iter() {
                let Some((v, i)) = (unsafe { (*ent.0).get_finished() }) else { continue };
                f(k, v, DepNodeIndex::from_u32(i));
            }
        }
    }
}

/// In-memory cache for queries whose key type only has one value (e.g. `()`).
/// The cache therefore only needs to store one query return value.
pub struct SingleCache<V> {
    entry: CacheEntry<V>,
}

impl<V> Default for SingleCache<V> {
    fn default() -> Self {
        SingleCache { entry: CacheEntry::empty() }
    }
}

impl<V: Copy> QueryCache for SingleCache<V> {
    type Key = ();
    type Value = V;

    #[inline(always)]
    fn lookup(&self, _key: ()) -> &CacheEntry<V> {
        &self.entry
    }

    fn for_each(&self, mut f: impl FnMut(Self::Key, &Self::Value, DepNodeIndex)) {
        if let Some((value, index)) = self.entry.get_finished() {
            f((), value, DepNodeIndex::from_u32(index))
        }
    }
}

/// In-memory cache for queries whose key is a [`DefId`].
///
/// Selects between one of two internal caches, depending on whether the key
/// is a local ID or foreign-crate ID.
pub struct DefIdCache<V> {
    /// Stores the local DefIds in a dense map. Local queries are much more often dense, so this is
    /// a win over hashing query keys at marginal memory cost (~5% at most) compared to FxHashMap.
    local: VecCache<DefIndex, V, DepNodeIndex>,
    foreign: DefaultCache<DefId, V>,
}

impl<V> Default for DefIdCache<V> {
    fn default() -> Self {
        DefIdCache { local: Default::default(), foreign: Default::default() }
    }
}

impl<V: Copy> QueryCache for DefIdCache<V> {
    type Key = DefId;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: DefId) -> &CacheEntry<V> {
        if key.krate == LOCAL_CRATE {
            self.local.lookup(key.index)
        } else {
            self.foreign.lookup(key)
        }
    }

    fn for_each(&self, mut f: impl FnMut(Self::Key, &Self::Value, DepNodeIndex)) {
        self.local.for_each(|index, value, dep_index| {
            f(DefId { krate: LOCAL_CRATE, index }, value, dep_index);
        });
        self.foreign.for_each(f);
    }
}

impl<K, V: Copy> QueryCache for VecCache<K, V, DepNodeIndex>
where
    K: Idx + QueryKey,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: K) -> &CacheEntry<V> {
        self.lookup(key)
    }

    #[inline(always)]
    fn for_each(&self, f: impl FnMut(K, &V, DepNodeIndex)) {
        self.for_each(f)
    }
}
