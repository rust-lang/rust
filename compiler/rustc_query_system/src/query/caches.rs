use crate::dep_graph::DepNodeIndex;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded::{self, Sharded};
use rustc_data_structures::sync::OnceLock;
use rustc_index::{Idx, IndexVec};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait CacheSelector<'tcx, V> {
    type Cache
    where
        V: Copy;
}

pub trait QueryCache: Sized {
    type Key: Hash + Eq + Copy + Debug;
    type Value: Copy;

    /// Checks if the query is already computed and in the cache.
    fn lookup(&self, key: &Self::Key) -> Option<(Self::Value, DepNodeIndex)>;

    fn complete(&self, key: Self::Key, value: Self::Value, index: DepNodeIndex);

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex));
}

pub struct DefaultCacheSelector<K>(PhantomData<K>);

impl<'tcx, K: Eq + Hash, V: 'tcx> CacheSelector<'tcx, V> for DefaultCacheSelector<K> {
    type Cache = DefaultCache<K, V>
    where
        V: Copy;
}

pub struct DefaultCache<K, V> {
    cache: Sharded<FxHashMap<K, (V, DepNodeIndex)>>,
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
        let result = lock.raw_entry().from_key_hashed_nocheck(key_hash, key);

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

pub struct SingleCacheSelector;

impl<'tcx, V: 'tcx> CacheSelector<'tcx, V> for SingleCacheSelector {
    type Cache = SingleCache<V>
    where
        V: Copy;
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

pub struct VecCacheSelector<K>(PhantomData<K>);

impl<'tcx, K: Idx, V: 'tcx> CacheSelector<'tcx, V> for VecCacheSelector<K> {
    type Cache = VecCache<K, V>
    where
        V: Copy;
}

pub struct VecCache<K: Idx, V> {
    cache: Sharded<IndexVec<K, Option<(V, DepNodeIndex)>>>,
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
        let lock = self.cache.lock_shard_by_hash(key.index() as u64);
        if let Some(Some(value)) = lock.get(*key) { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        let mut lock = self.cache.lock_shard_by_hash(key.index() as u64);
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        for shard in self.cache.lock_shards() {
            for (k, v) in shard.iter_enumerated() {
                if let Some(v) = v {
                    f(&k, &v.0, v.1);
                }
            }
        }
    }
}
