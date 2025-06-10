use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Mutex, OnceLock};

use rustc_data_structures::sharded::ShardedHashMap;
pub use rustc_data_structures::vec_cache::VecCache;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_index::Idx;
use rustc_span::def_id::{DefId, DefIndex};

use crate::dep_graph::DepNodeIndex;

/// Trait for types that serve as an in-memory cache for query results,
/// for a given key (argument) type and value (return) type.
///
/// Types implementing this trait are associated with actual key/value types
/// by the `Cache` associated type of the `rustc_middle::query::Key` trait.
pub trait QueryCache: Sized {
    type Key: Hash + Eq + Copy + Debug;
    type Value: Copy;

    /// Returns the cached value (and other information) associated with the
    /// given key, if it is present in the cache.
    fn lookup(&self, key: &Self::Key) -> Option<(Self::Value, DepNodeIndex)>;

    /// Start the job.
    ///
    /// After the first call to `to_unique_index`, the returned index is guaranteed to be
    /// uniquely bound to `key` until earlier of `drop(self)` or `complete(key, ...)`.
    fn to_unique_index(&self, key: &Self::Key) -> usize;

    /// Reverse the mapping of to_unique_index to a key.
    ///
    /// Will panic if called with an index that's not currently available.
    fn to_key(&self, idx: usize) -> Self::Key;

    /// Adds a key/value entry to this cache.
    ///
    /// Called by some part of the query system, after having obtained the
    /// value by executing the query or loading a cached value from disk.
    fn complete(&self, key: Self::Key, value: Self::Value, index: DepNodeIndex);

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex));
}

/// In-memory cache for queries whose keys aren't suitable for any of the
/// more specialized kinds of cache. Backed by a sharded hashmap.
pub struct DefaultCache<K, V> {
    cache: ShardedHashMap<K, (V, DepNodeIndex)>,
    active: Mutex<Vec<Option<K>>>,
}

impl<K, V> Default for DefaultCache<K, V> {
    fn default() -> Self {
        DefaultCache { cache: Default::default(), active: Default::default() }
    }
}

impl<K, V> QueryCache for DefaultCache<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        self.cache.get(key)
    }

    fn to_unique_index(&self, key: &Self::Key) -> usize {
        let mut guard = self.active.lock().unwrap();

        for (idx, slot) in guard.iter_mut().enumerate() {
            if let Some(k) = slot
                && k == key
            {
                // Return idx if we found the slot containing this key.
                return idx;
            } else if slot.is_none() {
                // If slot is empty, reuse it.
                *slot = Some(*key);
                return idx;
            }
        }

        // If no slot currently contains our key, add a new slot.
        let idx = guard.len();
        guard.push(Some(*key));
        return idx;
    }

    fn to_key(&self, idx: usize) -> Self::Key {
        let guard = self.active.lock().unwrap();
        guard[idx].expect("still present")
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        // We may be overwriting another value. This is all right, since the dep-graph
        // will check that the fingerprint matches.
        self.cache.insert(key, (value, index));

        // Make sure to do this second -- this ensures lookups return success prior to active
        // getting removed, helping avoiding assignment of multiple indices per logical key.
        let mut guard = self.active.lock().unwrap();
        for slot in guard.iter_mut() {
            if let Some(k) = slot
                && *k == key
            {
                *slot = None;
            }
        }
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        for shard in self.cache.lock_shards() {
            for (k, v) in shard.iter() {
                f(k, &v.0, v.1);
            }
        }
    }
}

/// In-memory cache for queries whose key type only has one value (e.g. `()`).
/// The cache therefore only needs to store one query return value.
pub struct SingleCache<V> {
    cache: OnceLock<(V, DepNodeIndex)>,
}

impl<V> Default for SingleCache<V> {
    fn default() -> Self {
        SingleCache { cache: OnceLock::new() }
    }
}

impl<V> QueryCache for SingleCache<V>
where
    V: Copy,
{
    type Key = ();
    type Value = V;

    #[inline(always)]
    fn lookup(&self, _key: &()) -> Option<(V, DepNodeIndex)> {
        self.cache.get().copied()
    }

    #[inline]
    fn complete(&self, _key: (), value: V, index: DepNodeIndex) {
        self.cache.set((value, index)).ok();
    }

    fn to_unique_index(&self, _: &Self::Key) -> usize {
        // SingleCache has a single key, so we can map directly to a constant.
        0
    }

    fn to_key(&self, _: usize) -> Self::Key {
        ()
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        if let Some(value) = self.cache.get() {
            f(&(), &value.0, value.1)
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

impl<V> QueryCache for DefIdCache<V>
where
    V: Copy,
{
    type Key = DefId;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &DefId) -> Option<(V, DepNodeIndex)> {
        if key.krate == LOCAL_CRATE {
            self.local.lookup(&key.index)
        } else {
            self.foreign.lookup(key)
        }
    }

    fn to_unique_index(&self, key: &Self::Key) -> usize {
        if key.krate == LOCAL_CRATE {
            // The local cache assigns keys based on allocated addresses in the backing VecCache.
            //
            // Those addresses are always at least 4-aligned (due to DepNodeIndex), so the low bit
            // can't be set. This means these don't overlap with the other cache given we |1 those
            // IDs. We check this with the assertion.
            let local_idx = self.local.to_unique_index(&key.index);
            assert!(local_idx & 1 == 0);
            local_idx
        } else {
            // Shifting is safe because DefaultCache uses only u32 for its indices, so we won't
            // overflow here.
            (self.foreign.to_unique_index(key) << 1) | 1
        }
    }

    fn to_key(&self, idx: usize) -> Self::Key {
        if idx & 1 == 0 {
            // local
            DefId::local(self.local.to_key(idx))
        } else {
            self.foreign.to_key(idx >> 1)
        }
    }

    #[inline]
    fn complete(&self, key: DefId, value: V, index: DepNodeIndex) {
        if key.krate == LOCAL_CRATE {
            self.local.complete(key.index, value, index)
        } else {
            self.foreign.complete(key, value, index)
        }
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        self.local.iter(&mut |key, value, index| {
            f(&DefId { krate: LOCAL_CRATE, index: *key }, value, index);
        });
        self.foreign.iter(f);
    }
}

impl<K, V> QueryCache for VecCache<K, V, DepNodeIndex>
where
    K: Idx + Eq + Hash + Copy + Debug,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        self.lookup(key)
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        self.complete(key, value, index)
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        self.iter(f)
    }

    fn to_unique_index(&self, key: &Self::Key) -> usize {
        self.to_slot_address(key)
    }

    fn to_key(&self, idx: usize) -> Self::Key {
        self.to_key(idx)
    }
}
