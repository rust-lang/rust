use crate::dep_graph::DepNodeIndex;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded;
#[cfg(parallel_compiler)]
use rustc_data_structures::sharded::Sharded;
use rustc_data_structures::sync::Lock;
use rustc_index::vec::{Idx, IndexVec};
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
    type Value: Copy + Debug;

    /// Checks if the query is already computed and in the cache.
    /// It returns the shard index and a lock guard to the shard,
    /// which will be used if the query is not in the cache and we need
    /// to compute it.
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
    #[cfg(parallel_compiler)]
    cache: Sharded<FxHashMap<K, (V, DepNodeIndex)>>,
    #[cfg(not(parallel_compiler))]
    cache: Lock<FxHashMap<K, (V, DepNodeIndex)>>,
}

impl<K, V> Default for DefaultCache<K, V> {
    fn default() -> Self {
        DefaultCache { cache: Default::default() }
    }
}

impl<K, V> QueryCache for DefaultCache<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Copy + Debug,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        let key_hash = sharded::make_hash(key);
        #[cfg(parallel_compiler)]
        let lock = self.cache.get_shard_by_hash(key_hash).lock();
        #[cfg(not(parallel_compiler))]
        let lock = self.cache.lock();
        let result = lock.raw_entry().from_key_hashed_nocheck(key_hash, key);

        if let Some((_, value)) = result { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        #[cfg(parallel_compiler)]
        let mut lock = self.cache.get_shard_by_value(&key).lock();
        #[cfg(not(parallel_compiler))]
        let mut lock = self.cache.lock();
        // We may be overwriting another value. This is all right, since the dep-graph
        // will check that the fingerprint matches.
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        #[cfg(parallel_compiler)]
        {
            let shards = self.cache.lock_shards();
            for shard in shards.iter() {
                for (k, v) in shard.iter() {
                    f(k, &v.0, v.1);
                }
            }
        }
        #[cfg(not(parallel_compiler))]
        {
            let map = self.cache.lock();
            for (k, v) in map.iter() {
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
    cache: Lock<Option<(V, DepNodeIndex)>>,
}

impl<V> Default for SingleCache<V> {
    fn default() -> Self {
        SingleCache { cache: Lock::new(None) }
    }
}

impl<V> QueryCache for SingleCache<V>
where
    V: Copy + Debug,
{
    type Key = ();
    type Value = V;

    #[inline(always)]
    fn lookup(&self, _key: &()) -> Option<(V, DepNodeIndex)> {
        *self.cache.lock()
    }

    #[inline]
    fn complete(&self, _key: (), value: V, index: DepNodeIndex) {
        *self.cache.lock() = Some((value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        self.cache.lock().as_ref().map(|value| f(&(), &value.0, value.1));
    }
}

pub struct VecCacheSelector<K>(PhantomData<K>);

impl<'tcx, K: Idx, V: 'tcx> CacheSelector<'tcx, V> for VecCacheSelector<K> {
    type Cache = VecCache<K, V>
    where
        V: Copy;
}

pub struct VecCache<K: Idx, V> {
    #[cfg(parallel_compiler)]
    cache: Sharded<IndexVec<K, Option<(V, DepNodeIndex)>>>,
    #[cfg(not(parallel_compiler))]
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
    V: Copy + Debug,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        #[cfg(parallel_compiler)]
        let lock = self.cache.get_shard_by_hash(key.index() as u64).lock();
        #[cfg(not(parallel_compiler))]
        let lock = self.cache.lock();
        if let Some(Some(value)) = lock.get(*key) { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        #[cfg(parallel_compiler)]
        let mut lock = self.cache.get_shard_by_hash(key.index() as u64).lock();
        #[cfg(not(parallel_compiler))]
        let mut lock = self.cache.lock();
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        #[cfg(parallel_compiler)]
        {
            let shards = self.cache.lock_shards();
            for shard in shards.iter() {
                for (k, v) in shard.iter_enumerated() {
                    if let Some(v) = v {
                        f(&k, &v.0, v.1);
                    }
                }
            }
        }
        #[cfg(not(parallel_compiler))]
        {
            let map = self.cache.lock();
            for (k, v) in map.iter_enumerated() {
                if let Some(v) = v {
                    f(&k, &v.0, v.1);
                }
            }
        }
    }
}
