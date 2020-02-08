use crate::dep_graph::DepNodeIndex;
use crate::ty::query::config::QueryAccessors;
use crate::ty::query::plumbing::{QueryLookup, QueryState, QueryStateShard};
use crate::ty::TyCtxt;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded::Sharded;
use rustc_hir::def_id::{DefId, DefIndex, LOCAL_CRATE};
use rustc_index::vec::IndexVec;
use std::cell::RefCell;
use std::default::Default;
use std::hash::Hash;
use std::marker::PhantomData;

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
    type Cache = DefaultCache<()>;
}

pub struct DefaultCache<D>(PhantomData<D>);

impl<D> Default for DefaultCache<D> {
    fn default() -> Self {
        DefaultCache(PhantomData)
    }
}

impl<D, K: Eq + Hash, V: Clone> QueryCache<K, V> for DefaultCache<D> {
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

#[cfg(parallel_compiler)]
pub type LocalDenseDefIdCacheSelector<V> = DefaultCache<V>;
#[cfg(not(parallel_compiler))]
pub type LocalDenseDefIdCacheSelector<V> = LocalDenseDefIdCache<V>;

pub struct LocalDenseDefIdCache<V> {
    local: RefCell<IndexVec<DefIndex, Option<(V, DepNodeIndex)>>>,
    other: DefaultCache<()>,
}

impl<V> Default for LocalDenseDefIdCache<V> {
    fn default() -> Self {
        LocalDenseDefIdCache { local: RefCell::new(IndexVec::new()), other: Default::default() }
    }
}

impl<V: Clone> QueryCache<DefId, V> for LocalDenseDefIdCache<V> {
    type Sharded = <DefaultCache<()> as QueryCache<DefId, V>>::Sharded;

    #[inline(always)]
    fn lookup<'tcx, R, GetCache, OnHit, OnMiss, Q>(
        &self,
        state: &'tcx QueryState<'tcx, Q>,
        get_cache: GetCache,
        key: DefId,
        on_hit: OnHit,
        on_miss: OnMiss,
    ) -> R
    where
        Q: QueryAccessors<'tcx>,
        GetCache: for<'a> Fn(&'a mut QueryStateShard<'tcx, Q>) -> &'a mut Self::Sharded,
        OnHit: FnOnce(&V, DepNodeIndex) -> R,
        OnMiss: FnOnce(DefId, QueryLookup<'tcx, Q>) -> R,
    {
        if key.krate == LOCAL_CRATE {
            let local = self.local.borrow();
            if let Some(result) = local.get(key.index).and_then(|v| v.as_ref()) {
                on_hit(&result.0, result.1)
            } else {
                drop(local);
                let lookup = state.get_lookup(&key);
                on_miss(key, lookup)
            }
        } else {
            self.other.lookup(state, get_cache, key, on_hit, on_miss)
        }
    }

    #[inline]
    fn complete(
        &self,
        tcx: TyCtxt<'tcx>,
        lock_sharded_storage: &mut Self::Sharded,
        key: DefId,
        value: V,
        index: DepNodeIndex,
    ) {
        if key.krate == LOCAL_CRATE {
            let mut local = self.local.borrow_mut();
            if local.raw.capacity() == 0 {
                *local = IndexVec::from_elem_n(None, tcx.hir().definitions().def_index_count());
            }
            local[key.index] = Some((value, index));
        } else {
            self.other.complete(tcx, lock_sharded_storage, key, value, index);
        }
    }

    fn iter<R, L>(
        &self,
        shards: &Sharded<L>,
        get_shard: impl Fn(&mut L) -> &mut Self::Sharded,
        f: impl for<'a> FnOnce(Box<dyn Iterator<Item = (&'a DefId, &'a V, DepNodeIndex)> + 'a>) -> R,
    ) -> R {
        let local = self.local.borrow();
        let local: Vec<(DefId, &V, DepNodeIndex)> = local
            .iter_enumerated()
            .filter_map(|(i, e)| e.as_ref().map(|e| (DefId::local(i), &e.0, e.1)))
            .collect();
        self.other.iter(shards, get_shard, |results| {
            f(Box::new(results.chain(local.iter().map(|(id, v, i)| (id, *v, *i)))))
        })
    }
}
