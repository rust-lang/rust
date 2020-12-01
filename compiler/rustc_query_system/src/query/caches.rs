use crate::dep_graph::DepNodeIndex;
use crate::query::plumbing::{QueryLookup, QueryState};

use rustc_arena::TypedArena;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded::Sharded;
use rustc_data_structures::sync::WorkerLocal;
use std::default::Default;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait CacheSelector<K, V> {
    type Cache;
}

pub trait QueryStorage: Default {
    type Value;
    type Stored: Clone;

    /// Store a value without putting it in the cache.
    /// This is meant to be used with cycle errors.
    fn store_nocache(&self, value: Self::Value) -> Self::Stored;
}

pub trait QueryCache: QueryStorage {
    type Key: Hash + Eq + Clone + Debug;
    type Sharded: Default;

    /// Checks if the query is already computed and in the cache.
    /// It returns the shard index and a lock guard to the shard,
    /// which will be used if the query is not in the cache and we need
    /// to compute it.
    fn lookup<D, Q, R, OnHit, OnMiss>(
        &self,
        state: &QueryState<D, Q, Self>,
        key: Self::Key,
        // `on_hit` can be called while holding a lock to the query state shard.
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        OnHit: FnOnce(&Self::Stored, DepNodeIndex) -> R,
        OnMiss: FnOnce(Self::Key, QueryLookup<'_, D, Q, Self::Key, Self::Sharded>) -> R;

    fn complete(
        &self,
        lock_sharded_storage: &mut Self::Sharded,
        key: Self::Key,
        value: Self::Value,
        index: DepNodeIndex,
    ) -> Self::Stored;

    fn iter<R, L>(
        &self,
        shards: &Sharded<L>,
        get_shard: impl Fn(&mut L) -> &mut Self::Sharded,
        f: impl for<'a> FnOnce(
            Box<dyn Iterator<Item = (&'a Self::Key, &'a Self::Value, DepNodeIndex)> + 'a>,
        ) -> R,
    ) -> R;
}

pub struct DefaultCacheSelector;

impl<K: Eq + Hash, V: Clone> CacheSelector<K, V> for DefaultCacheSelector {
    type Cache = DefaultCache<K, V>;
}

pub struct DefaultCache<K, V>(PhantomData<(K, V)>);

impl<K, V> Default for DefaultCache<K, V> {
    fn default() -> Self {
        DefaultCache(PhantomData)
    }
}

impl<K: Eq + Hash, V: Clone> QueryStorage for DefaultCache<K, V> {
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
    V: Clone,
{
    type Key = K;
    type Sharded = FxHashMap<K, (V, DepNodeIndex)>;

    #[inline(always)]
    fn lookup<D, Q, R, OnHit, OnMiss>(
        &self,
        state: &QueryState<D, Q, Self>,
        key: K,
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        OnHit: FnOnce(&V, DepNodeIndex) -> R,
        OnMiss: FnOnce(K, QueryLookup<'_, D, Q, K, Self::Sharded>) -> R,
    {
        let mut lookup = state.get_lookup(&key);
        let lock = &mut *lookup.lock;

        let result = lock.cache.raw_entry().from_key_hashed_nocheck(lookup.key_hash, &key);

        if let Some((_, value)) = result { on_hit(&value.0, value.1) } else { on_miss(key, lookup) }
    }

    #[inline]
    fn complete(
        &self,
        lock_sharded_storage: &mut Self::Sharded,
        key: K,
        value: V,
        index: DepNodeIndex,
    ) -> Self::Stored {
        lock_sharded_storage.insert(key, (value.clone(), index));
        value
    }

    fn iter<R, L>(
        &self,
        shards: &Sharded<L>,
        get_shard: impl Fn(&mut L) -> &mut Self::Sharded,
        f: impl for<'a> FnOnce(Box<dyn Iterator<Item = (&'a K, &'a V, DepNodeIndex)> + 'a>) -> R,
    ) -> R {
        let mut shards = shards.lock_shards();
        let mut shards: Vec<_> = shards.iter_mut().map(|shard| get_shard(shard)).collect();
        let results = shards.iter_mut().flat_map(|shard| shard.iter()).map(|(k, v)| (k, &v.0, v.1));
        f(Box::new(results))
    }
}

pub struct ArenaCacheSelector<'tcx>(PhantomData<&'tcx ()>);

impl<'tcx, K: Eq + Hash, V: 'tcx> CacheSelector<K, V> for ArenaCacheSelector<'tcx> {
    type Cache = ArenaCache<'tcx, K, V>;
}

pub struct ArenaCache<'tcx, K, V> {
    arena: WorkerLocal<TypedArena<(V, DepNodeIndex)>>,
    phantom: PhantomData<(K, &'tcx V)>,
}

impl<'tcx, K, V> Default for ArenaCache<'tcx, K, V> {
    fn default() -> Self {
        ArenaCache { arena: WorkerLocal::new(|_| TypedArena::default()), phantom: PhantomData }
    }
}

impl<'tcx, K: Eq + Hash, V: 'tcx> QueryStorage for ArenaCache<'tcx, K, V> {
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
{
    type Key = K;
    type Sharded = FxHashMap<K, &'tcx (V, DepNodeIndex)>;

    #[inline(always)]
    fn lookup<D, Q, R, OnHit, OnMiss>(
        &self,
        state: &QueryState<D, Q, Self>,
        key: K,
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        OnHit: FnOnce(&&'tcx V, DepNodeIndex) -> R,
        OnMiss: FnOnce(K, QueryLookup<'_, D, Q, K, Self::Sharded>) -> R,
    {
        let mut lookup = state.get_lookup(&key);
        let lock = &mut *lookup.lock;

        let result = lock.cache.raw_entry().from_key_hashed_nocheck(lookup.key_hash, &key);

        if let Some((_, value)) = result {
            on_hit(&&value.0, value.1)
        } else {
            on_miss(key, lookup)
        }
    }

    #[inline]
    fn complete(
        &self,
        lock_sharded_storage: &mut Self::Sharded,
        key: K,
        value: V,
        index: DepNodeIndex,
    ) -> Self::Stored {
        let value = self.arena.alloc((value, index));
        let value = unsafe { &*(value as *const _) };
        lock_sharded_storage.insert(key, value);
        &value.0
    }

    fn iter<R, L>(
        &self,
        shards: &Sharded<L>,
        get_shard: impl Fn(&mut L) -> &mut Self::Sharded,
        f: impl for<'a> FnOnce(Box<dyn Iterator<Item = (&'a K, &'a V, DepNodeIndex)> + 'a>) -> R,
    ) -> R {
        let mut shards = shards.lock_shards();
        let mut shards: Vec<_> = shards.iter_mut().map(|shard| get_shard(shard)).collect();
        let results = shards.iter_mut().flat_map(|shard| shard.iter()).map(|(k, v)| (k, &v.0, v.1));
        f(Box::new(results))
    }
}
