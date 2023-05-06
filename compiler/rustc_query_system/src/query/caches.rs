use crate::dep_graph::DepNodeIndex;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded;
use rustc_data_structures::sync::{LockLike, SLock};
use rustc_index::{Idx, IndexVec};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait CacheSelector<'tcx, V, L> {
    type Cache
    where
        V: Copy,
        L: SLock;
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

impl<'tcx, K: Eq + Hash, V: 'tcx, L: SLock> CacheSelector<'tcx, V, L> for DefaultCacheSelector<K> {
    type Cache = DefaultCache<K, V, L>
    where
        V: Copy;
}

pub struct DefaultCache<K, V, L: SLock> {
    cache: L::Lock<FxHashMap<K, (V, DepNodeIndex)>>,
}

impl<K, V, L: SLock> Default for DefaultCache<K, V, L> {
    fn default() -> Self {
        DefaultCache { cache: L::Lock::new(FxHashMap::default()) }
    }
}

impl<K, V, L: SLock> QueryCache for DefaultCache<K, V, L>
where
    K: Eq + Hash + Copy + Debug,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        let key_hash = sharded::make_hash(key);

        let lock = self.cache.lock();

        let result = lock.raw_entry().from_key_hashed_nocheck(key_hash, key);

        if let Some((_, value)) = result { Some(*value) } else { None }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        let mut lock = self.cache.lock();

        // We may be overwriting another value. This is all right, since the dep-graph
        // will check that the fingerprint matches.
        lock.insert(key, (value, index));
    }

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        let shards = self.cache.lock();
        for (k, v) in shards.iter() {
            f(k, &v.0, v.1);
        }
    }
}

pub struct SingleCacheSelector;

impl<'tcx, V: 'tcx, L: SLock> CacheSelector<'tcx, V, L> for SingleCacheSelector {
    type Cache = SingleCache<V, L>
    where
        V: Copy;
}

pub struct SingleCache<V, L: SLock> {
    cache: L::Lock<Option<(V, DepNodeIndex)>>,
}

impl<V, L: SLock> Default for SingleCache<V, L> {
    fn default() -> Self {
        SingleCache { cache: L::Lock::new(None) }
    }
}

impl<V, L: SLock> QueryCache for SingleCache<V, L>
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

impl<'tcx, K: Idx, V: 'tcx, L: SLock> CacheSelector<'tcx, V, L> for VecCacheSelector<K> {
    type Cache = VecCache<K, V, L>
    where
        V: Copy;
}

pub struct VecCache<K: Idx, V, L: SLock> {
    cache: L::Lock<IndexVec<K, Option<(V, DepNodeIndex)>>>,
}

impl<K: Idx, V, L: SLock> Default for VecCache<K, V, L> {
    fn default() -> Self {
        VecCache { cache: L::Lock::new(IndexVec::default()) }
    }
}

impl<K, V, L: SLock> QueryCache for VecCache<K, V, L>
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
        let shards = self.cache.lock();
        for (k, v) in shards.iter_enumerated() {
            if let Some(v) = v {
                f(&k, &v.0, v.1);
            }
        }
    }
}
