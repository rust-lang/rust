use crate::dep_graph::DepNodeIndex;
use crate::ty::query::config::QueryAccessors;
use crate::ty::query::plumbing::{QueryLookup, QueryState, QueryStateShard};
use crate::ty::TyCtxt;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded::Sharded;
use std::default::Default;
use std::hash::Hash;

pub(crate) trait CacheSelector<K, V> {
    type Cache: QueryCache<K, V>;
}

pub(crate) trait QueryCache<K, V>: Default {
    type Sharded: Default;

    /// Checks if the query is already computed and in the cache.
    /// It returns the shard index and a lock guard to the shard,
    /// which will be used if the query is not in the cache and we need
    /// to compute it.
    fn lookup<'tcx, R, GetCache, OnHit, OnMiss, Q>(
        &self,
        state: &'tcx QueryState<'tcx, Q>,
        get_cache: GetCache,
        key: K,
        // `on_hit` can be called while holding a lock to the query state shard.
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        Q: QueryAccessors<'tcx>,
        GetCache: for<'a> Fn(&'a mut QueryStateShard<'tcx, Q>) -> &'a mut Self::Sharded,
        OnHit: FnOnce(&V, DepNodeIndex) -> R,
        OnMiss: FnOnce(K, QueryLookup<'tcx, Q>) -> R;

    fn complete(
        &self,
        tcx: TyCtxt<'tcx>,
        lock_sharded_storage: &mut Self::Sharded,
        key: K,
        value: V,
        index: DepNodeIndex,
    );

    fn iter<R, L>(
        &self,
        shards: &Sharded<L>,
        get_shard: impl Fn(&mut L) -> &mut Self::Sharded,
        f: impl for<'a> FnOnce(Box<dyn Iterator<Item = (&'a K, &'a V, DepNodeIndex)> + 'a>) -> R,
    ) -> R;
}

pub struct DefaultCacheSelector;

impl<K: Eq + Hash, V: Clone> CacheSelector<K, V> for DefaultCacheSelector {
    type Cache = DefaultCache;
}

#[derive(Default)]
pub struct DefaultCache;

impl<K: Eq + Hash, V: Clone> QueryCache<K, V> for DefaultCache {
    type Sharded = FxHashMap<K, (V, DepNodeIndex)>;

    #[inline(always)]
    fn lookup<'tcx, R, GetCache, OnHit, OnMiss, Q>(
        &self,
        state: &'tcx QueryState<'tcx, Q>,
        get_cache: GetCache,
        key: K,
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        Q: QueryAccessors<'tcx>,
        GetCache: for<'a> Fn(&'a mut QueryStateShard<'tcx, Q>) -> &'a mut Self::Sharded,
        OnHit: FnOnce(&V, DepNodeIndex) -> R,
        OnMiss: FnOnce(K, QueryLookup<'tcx, Q>) -> R,
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
