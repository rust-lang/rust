use crate::dep_graph::DepNodeIndex;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded;
use rustc_data_structures::sharded::{Shard, ShardImpl};
use rustc_data_structures::sync::LockLike;
use rustc_index::{Idx, IndexVec};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait CacheSelector<'tcx, V, S> {
    type Cache
    where
        V: Copy,
        S: Shard;
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

impl<'tcx, K: Eq + Hash, V: 'tcx, S: Shard> CacheSelector<'tcx, V, S> for DefaultCacheSelector<K> {
    type Cache = DefaultCache<K, V, S>
    where
        V: Copy;
}

pub struct DefaultCache<K, V, S: Shard> {
    cache: S::Impl<FxHashMap<K, (V, DepNodeIndex)>>,
}

impl<K, V, S: Shard> Default for DefaultCache<K, V, S> {
    fn default() -> Self {
        DefaultCache { cache: S::Impl::new(|| FxHashMap::default()) }
    }
}

impl<K, V, S: Shard> QueryCache for DefaultCache<K, V, S>
where
    K: Eq + Hash + Copy + Debug,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        let key_hash = sharded::make_hash(key);

        let lock = self.cache.get_shard_by_hash(key_hash).lock();

        let result = lock.raw_entry().from_key_hashed_nocheck(key_hash, key);

        if let Some((_, value)) = result { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        let mut lock = self.cache.get_shard_by_value(&key).lock();

        // We may be overwriting another value. This is all right, since the dep-graph
        // will check that the fingerprint matches.
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        let shards = self.cache.lock_shards();
        for shard in shards.into_iter() {
            for (k, v) in shard.iter() {
                f(k, &v.0, v.1);
            }
        }
    }
}

pub struct SingleCacheSelector;

impl<'tcx, V: 'tcx, S: Shard> CacheSelector<'tcx, V, S> for SingleCacheSelector {
    type Cache = SingleCache<V, S>
    where
        V: Copy;
}

pub struct SingleCache<V, S: Shard> {
    cache: <S::Impl<Option<(V, DepNodeIndex)>> as ShardImpl<Option<(V, DepNodeIndex)>>>::Lock,
}

impl<V, S: Shard> Default for SingleCache<V, S> {
    fn default() -> Self {
        SingleCache { cache: <S::Impl<Option<(V, DepNodeIndex)>> as ShardImpl<Option<(V, DepNodeIndex)>>>::Lock::new(None) }
    }
}

impl<V, S: Shard> QueryCache for SingleCache<V, S>
where
    V: Copy,
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
        if let Some(value) = self.cache.lock().as_ref() {
            f(&(), &value.0, value.1)
        }
    }
}

pub struct VecCacheSelector<K>(PhantomData<K>);

impl<'tcx, K: Idx, V: 'tcx, S: Shard> CacheSelector<'tcx, V, S> for VecCacheSelector<K> {
    type Cache = VecCache<K, V, S>
    where
        V: Copy;
}

pub struct VecCache<K: Idx, V, S: Shard> {
    cache: S::Impl<IndexVec<K, Option<(V, DepNodeIndex)>>>,
}

impl<K: Idx, V, S: Shard> Default for VecCache<K, V, S> {
    fn default() -> Self {
        VecCache { cache: S::Impl::new(|| IndexVec::default()) }
    }
}

impl<K, V, S: Shard> QueryCache for VecCache<K, V, S>
where
    K: Eq + Idx + Copy + Debug,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        let lock = self.cache.get_shard_by_hash(key.index() as u64).lock();

        if let Some(Some(value)) = lock.get(*key) { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        let mut lock = self.cache.get_shard_by_hash(key.index() as u64).lock();

        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        let shards = self.cache.lock_shards();
        for shard in shards.iter() {
            for (k, v) in shard.iter_enumerated() {
                if let Some(v) = v {
                    f(&k, &v.0, v.1);
                }
            }
        }
    }
}
