use std::fmt::Debug;
use std::hash::Hash;

use rustc_data_structures::fx::{FxIndexMap, IndexRawEntryApiV1};
use rustc_data_structures::sharded::{self, Sharded};
use rustc_data_structures::sync::{Lock, OnceLock};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_index::{Idx, IndexVec};
use rustc_span::def_id::{DefId, DefIndex};

use crate::dep_graph::DepNodeIndex;

pub trait QueryCache: Sized {
    type Key: Hash + Eq + Copy + Debug;
    type Value: Copy;

    /// Checks if the query is already computed and in the cache.
    fn lookup(&self, key: &Self::Key) -> Option<(Self::Value, DepNodeIndex)>;

    fn complete(&self, key: Self::Key, value: Self::Value, index: DepNodeIndex);

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex));
}

pub struct DefaultCache<K, V> {
    cache: Sharded<FxIndexMap<K, (V, DepNodeIndex)>>,
}

impl<K, V> Default for DefaultCache<K, V> {
    fn default() -> Self {
        DefaultCache { cache: Default::default() }
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
        let key_hash = sharded::make_hash(key);
        let lock = self.cache.lock_shard_by_hash(key_hash);
        let result = lock.raw_entry_v1().from_key_hashed_nocheck(key_hash, key);

        if let Some((_, value)) = result { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        let mut lock = self.cache.lock_shard_by_value(&key);
        // We may be overwriting another value. This is all right, since the dep-graph
        // will check that the fingerprint matches.
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        for shard in self.cache.lock_shards() {
            for (k, v) in shard.iter() {
                f(k, &v.0, v.1);
            }
        }
    }
}

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

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        if let Some(value) = self.cache.get() {
            f(&(), &value.0, value.1)
        }
    }
}

pub struct VecCache<K: Idx, V> {
    cache: Lock<IndexVec<K, Option<(V, DepNodeIndex)>>>,
}

impl<K: Idx, V> Default for VecCache<K, V> {
    fn default() -> Self {
        VecCache { cache: Default::default() }
    }
}

impl<K, V> QueryCache for VecCache<K, V>
where
    K: Eq + Idx + Copy + Debug,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        let lock = self.cache.lock();
        if let Some(Some(value)) = lock.get(*key) { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        let mut lock = self.cache.lock();
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        for (k, v) in self.cache.lock().iter_enumerated() {
            if let Some(v) = v {
                f(&k, &v.0, v.1);
            }
        }
    }
}

pub struct DefIdCache<V> {
    /// Stores the local DefIds in a dense map. Local queries are much more often dense, so this is
    /// a win over hashing query keys at marginal memory cost (~5% at most) compared to FxHashMap.
    ///
    /// The second element of the tuple is the set of keys actually present in the IndexVec, used
    /// for faster iteration in `iter()`.
    local: Lock<(IndexVec<DefIndex, Option<(V, DepNodeIndex)>>, Vec<DefIndex>)>,
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
            let cache = self.local.lock();
            cache.0.get(key.index).and_then(|v| *v)
        } else {
            self.foreign.lookup(key)
        }
    }

    #[inline]
    fn complete(&self, key: DefId, value: V, index: DepNodeIndex) {
        if key.krate == LOCAL_CRATE {
            let mut cache = self.local.lock();
            let (cache, present) = &mut *cache;
            let slot = cache.ensure_contains_elem(key.index, Default::default);
            if slot.is_none() {
                // FIXME: Only store the present set when running in incremental mode. `iter` is not
                // used outside of saving caches to disk and self-profile.
                present.push(key.index);
            }
            *slot = Some((value, index));
        } else {
            self.foreign.complete(key, value, index)
        }
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        let guard = self.local.lock();
        let (cache, present) = &*guard;
        for &idx in present.iter() {
            let value = cache[idx].unwrap();
            f(&DefId { krate: LOCAL_CRATE, index: idx }, &value.0, value.1);
        }
        self.foreign.iter(f);
    }
}
