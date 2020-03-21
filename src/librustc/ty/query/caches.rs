use crate::dep_graph::DepNodeIndex;
use crate::ty::query::plumbing::{QueryLookup, QueryState, QueryStateShard};
use crate::ty::TyCtxt;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded::Sharded;
use std::default::Default;
use std::hash::Hash;
use std::marker::PhantomData;

pub(crate) trait CacheSelector<K, V> {
    type Cache: QueryCache<Key = K, Value = V>;
}

pub(crate) trait QueryCache: Default {
    type Key;
    type Value;
    type Sharded: Default;

    /// Checks if the query is already computed and in the cache.
    /// It returns the shard index and a lock guard to the shard,
    /// which will be used if the query is not in the cache and we need
    /// to compute it.
    fn lookup<'tcx, R, GetCache, OnHit, OnMiss>(
        &self,
        state: &'tcx QueryState<'tcx, Self>,
        get_cache: GetCache,
        key: Self::Key,
        // `on_hit` can be called while holding a lock to the query state shard.
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        GetCache: for<'a> Fn(
            &'a mut QueryStateShard<'tcx, Self::Key, Self::Sharded>,
        ) -> &'a mut Self::Sharded,
        OnHit: FnOnce(&Self::Value, DepNodeIndex) -> R,
        OnMiss: FnOnce(Self::Key, QueryLookup<'tcx, Self::Key, Self::Sharded>) -> R;

    fn complete(
        &self,
        tcx: TyCtxt<'tcx>,
        lock_sharded_storage: &mut Self::Sharded,
        key: Self::Key,
        value: Self::Value,
        index: DepNodeIndex,
    );

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

impl<K: Eq + Hash, V: Clone> QueryCache for DefaultCache<K, V> {
    type Key = K;
    type Value = V;
    type Sharded = FxHashMap<K, (V, DepNodeIndex)>;

    #[inline(always)]
    fn lookup<'tcx, R, GetCache, OnHit, OnMiss>(
        &self,
        state: &'tcx QueryState<'tcx, Self>,
        get_cache: GetCache,
        key: K,
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        GetCache:
            for<'a> Fn(&'a mut QueryStateShard<'tcx, K, Self::Sharded>) -> &'a mut Self::Sharded,
        OnHit: FnOnce(&V, DepNodeIndex) -> R,
        OnMiss: FnOnce(K, QueryLookup<'tcx, K, Self::Sharded>) -> R,
    {
        let mut lookup = state.get_lookup(&key);
        let lock = &mut *lookup.lock;

        let result = get_cache(lock).raw_entry().from_key_hashed_nocheck(lookup.key_hash, &key);

        if let Some((_, value)) = result { on_hit(&value.0, value.1) } else { on_miss(key, lookup) }
    }

    #[inline]
    fn complete(
        &self,
        _: TyCtxt<'tcx>,
        lock_sharded_storage: &mut Self::Sharded,
        key: K,
        value: V,
        index: DepNodeIndex,
    ) {
        lock_sharded_storage.insert(key, (value, index));
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
