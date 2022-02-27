use crate::dep_graph::DepNodeIndex;

use rustc_arena::TypedArena;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded;
#[cfg(parallel_compiler)]
use rustc_data_structures::sharded::Sharded;
#[cfg(not(parallel_compiler))]
use rustc_data_structures::sync::Lock;
use rustc_data_structures::sync::WorkerLocal;
use std::default::Default;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait CacheSelector<K, V> {
    type Cache;
}

pub trait QueryStorage {
    type Value: Debug;
    type Stored: Clone;

    /// Store a value without putting it in the cache.
    /// This is meant to be used with cycle errors.
    fn store_nocache(&self, value: Self::Value) -> Self::Stored;
}

pub trait QueryCache: QueryStorage + Sized {
    type Key: Hash + Eq + Clone + Debug;

    /// Checks if the query is already computed and in the cache.
    /// It returns the shard index and a lock guard to the shard,
    /// which will be used if the query is not in the cache and we need
    /// to compute it.
    fn lookup<R, OnHit>(
        &self,
        key: &Self::Key,
        // `on_hit` can be called while holding a lock to the query state shard.
        on_hit: OnHit,
    ) -> Result<R, ()>
    where
        OnHit: FnOnce(&Self::Stored, DepNodeIndex) -> R;

    fn complete(&self, key: Self::Key, value: Self::Value, index: DepNodeIndex) -> Self::Stored;

    fn iter(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex));
}

pub struct DefaultCacheSelector;

impl<K: Eq + Hash, V: Clone> CacheSelector<K, V> for DefaultCacheSelector {
    type Cache = DefaultCache<K, V>;
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

impl<K: Eq + Hash, V: Clone + Debug> QueryStorage for DefaultCache<K, V> {
    type Value = V;
    type Stored = V;

    #[inline]
    fn store_nocache(&self, value: Self::Value) -> Self::Stored {
        // We have no dedicated storage
        value
    }
}

impl<K, V> QueryCache for DefaultCache<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Clone + Debug,
{
    type Key = K;

    #[inline(always)]
    fn lookup<R, OnHit>(&self, key: &K, on_hit: OnHit) -> Result<R, ()>
    where
        OnHit: FnOnce(&V, DepNodeIndex) -> R,
    {
        let key_hash = sharded::make_hash(key);
        #[cfg(parallel_compiler)]
        let lock = self.cache.get_shard_by_hash(key_hash).lock();
        #[cfg(not(parallel_compiler))]
        let lock = self.cache.lock();
        let result = lock.raw_entry().from_key_hashed_nocheck(key_hash, key);

        if let Some((_, value)) = result {
            let hit_result = on_hit(&value.0, value.1);
            Ok(hit_result)
        } else {
            Err(())
        }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) -> Self::Stored {
        #[cfg(parallel_compiler)]
        let mut lock = self.cache.get_shard_by_value(&key).lock();
        #[cfg(not(parallel_compiler))]
        let mut lock = self.cache.lock();
        lock.insert(key, (value.clone(), index));
        value
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

pub struct ArenaCacheSelector<'tcx>(PhantomData<&'tcx ()>);

impl<'tcx, K: Eq + Hash, V: 'tcx> CacheSelector<K, V> for ArenaCacheSelector<'tcx> {
    type Cache = ArenaCache<'tcx, K, V>;
}

pub struct ArenaCache<'tcx, K, V> {
    arena: WorkerLocal<TypedArena<(V, DepNodeIndex)>>,
    #[cfg(parallel_compiler)]
    cache: Sharded<FxHashMap<K, &'tcx (V, DepNodeIndex)>>,
    #[cfg(not(parallel_compiler))]
    cache: Lock<FxHashMap<K, &'tcx (V, DepNodeIndex)>>,
}

impl<'tcx, K, V> Default for ArenaCache<'tcx, K, V> {
    fn default() -> Self {
        ArenaCache { arena: WorkerLocal::new(|_| TypedArena::default()), cache: Default::default() }
    }
}

impl<'tcx, K: Eq + Hash, V: Debug + 'tcx> QueryStorage for ArenaCache<'tcx, K, V> {
    type Value = V;
    type Stored = &'tcx V;

    #[inline]
    fn store_nocache(&self, value: Self::Value) -> Self::Stored {
        let value = self.arena.alloc((value, DepNodeIndex::INVALID));
        let value = unsafe { &*(&value.0 as *const _) };
        &value
    }
}

impl<'tcx, K, V: 'tcx> QueryCache for ArenaCache<'tcx, K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Debug,
{
    type Key = K;

    #[inline(always)]
    fn lookup<R, OnHit>(&self, key: &K, on_hit: OnHit) -> Result<R, ()>
    where
        OnHit: FnOnce(&&'tcx V, DepNodeIndex) -> R,
    {
        let key_hash = sharded::make_hash(key);
        #[cfg(parallel_compiler)]
        let lock = self.cache.get_shard_by_hash(key_hash).lock();
        #[cfg(not(parallel_compiler))]
        let lock = self.cache.lock();
        let result = lock.raw_entry().from_key_hashed_nocheck(key_hash, key);

        if let Some((_, value)) = result {
            let hit_result = on_hit(&&value.0, value.1);
            Ok(hit_result)
        } else {
            Err(())
        }
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) -> Self::Stored {
        let value = self.arena.alloc((value, index));
        let value = unsafe { &*(value as *const _) };
        #[cfg(parallel_compiler)]
        let mut lock = self.cache.get_shard_by_value(&key).lock();
        #[cfg(not(parallel_compiler))]
        let mut lock = self.cache.lock();
        lock.insert(key, value);
        &value.0
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
