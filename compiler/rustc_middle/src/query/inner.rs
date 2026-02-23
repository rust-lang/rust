//! Helper functions that serve as the immediate implementation of
//! `tcx.$query(..)` and its variations.

use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::dep_graph;
use crate::dep_graph::{DepKind, DepNodeKey};
use crate::query::erase::{self, Erasable, Erased};
use crate::query::plumbing::QueryVTable;
use crate::query::{QueryCache, QueryMode};
use crate::ty::TyCtxt;

/// Checks whether there is already a value for this key in the in-memory
/// query cache, returning that value if present.
///
/// (Also performs some associated bookkeeping, if a value was found.)
#[inline(always)]
fn try_get_cached<'tcx, C>(tcx: TyCtxt<'tcx>, cache: &C, key: &C::Key) -> Option<C::Value>
where
    C: QueryCache,
{
    match cache.lookup(key) {
        Some((value, index)) => {
            tcx.prof.query_cache_hit(index.into());
            tcx.dep_graph.read_index(index);
            Some(value)
        }
        None => None,
    }
}

/// Shared implementation of `tcx.$query(..)` and `tcx.at(span).$query(..)`
/// for all queries.
#[inline(always)]
pub(crate) fn query_get_at<'tcx, Cache>(
    tcx: TyCtxt<'tcx>,
    execute_query: fn(TyCtxt<'tcx>, Span, Cache::Key, QueryMode) -> Option<Cache::Value>,
    query_cache: &Cache,
    span: Span,
    key: Cache::Key,
) -> Cache::Value
where
    Cache: QueryCache,
{
    match try_get_cached(tcx, query_cache, &key) {
        Some(value) => value,
        None => execute_query(tcx, span, key, QueryMode::Get).unwrap(),
    }
}

/// Shared implementation of `tcx.ensure_ok().$query(..)` for most queries,
/// and `tcx.ensure_done().$query(..)` for all queries.
#[inline]
pub(crate) fn query_ensure<'tcx, Cache>(
    tcx: TyCtxt<'tcx>,
    execute_query: fn(TyCtxt<'tcx>, Span, Cache::Key, QueryMode) -> Option<Cache::Value>,
    query_cache: &Cache,
    key: Cache::Key,
    check_cache: bool,
) where
    Cache: QueryCache,
{
    if try_get_cached(tcx, query_cache, &key).is_none() {
        execute_query(tcx, DUMMY_SP, key, QueryMode::Ensure { check_cache });
    }
}

/// Shared implementation of `tcx.ensure_ok().$query(..)` for queries that
/// have the `return_result_from_ensure_ok` modifier.
#[inline]
pub(crate) fn query_ensure_error_guaranteed<'tcx, Cache, T>(
    tcx: TyCtxt<'tcx>,
    execute_query: fn(TyCtxt<'tcx>, Span, Cache::Key, QueryMode) -> Option<Cache::Value>,
    query_cache: &Cache,
    key: Cache::Key,
    check_cache: bool,
) -> Result<(), ErrorGuaranteed>
where
    Cache: QueryCache<Value = Erased<Result<T, ErrorGuaranteed>>>,
    Result<T, ErrorGuaranteed>: Erasable,
{
    if let Some(res) = try_get_cached(tcx, query_cache, &key) {
        erase::restore_val(res).map(drop)
    } else {
        execute_query(tcx, DUMMY_SP, key, QueryMode::Ensure { check_cache })
            .map(erase::restore_val)
            .map(|res| res.map(drop))
            // Either we actually executed the query, which means we got a full `Result`,
            // or we can just assume the query succeeded, because it was green in the
            // incremental cache. If it is green, that means that the previous compilation
            // that wrote to the incremental cache compiles successfully. That is only
            // possible if the cache entry was `Ok(())`, so we emit that here, without
            // actually encoding the `Result` in the cache or loading it from there.
            .unwrap_or(Ok(()))
    }
}

/// Common implementation of query feeding, used by `define_feedable!`.
pub(crate) fn query_feed<'tcx, Cache>(
    tcx: TyCtxt<'tcx>,
    dep_kind: DepKind,
    query_vtable: &QueryVTable<'tcx, Cache>,
    cache: &Cache,
    key: Cache::Key,
    value: Cache::Value,
) where
    Cache: QueryCache,
    Cache::Key: DepNodeKey<'tcx>,
{
    let format_value = query_vtable.format_value;

    // Check whether the in-memory cache already has a value for this key.
    match try_get_cached(tcx, cache, &key) {
        Some(old) => {
            // The query already has a cached value for this key.
            // That's OK if both values are the same, i.e. they have the same hash,
            // so now we check their hashes.
            if let Some(hasher_fn) = query_vtable.hash_result {
                let (old_hash, value_hash) = tcx.with_stable_hashing_context(|ref mut hcx| {
                    (hasher_fn(hcx, &old), hasher_fn(hcx, &value))
                });
                if old_hash != value_hash {
                    // We have an inconsistency. This can happen if one of the two
                    // results is tainted by errors. In this case, delay a bug to
                    // ensure compilation is doomed, and keep the `old` value.
                    tcx.dcx().delayed_bug(format!(
                        "Trying to feed an already recorded value for query {dep_kind:?} key={key:?}:\n\
                        old value: {old}\nnew value: {value}",
                        old = format_value(&old),
                        value = format_value(&value),
                    ));
                }
            } else {
                // The query is `no_hash`, so we have no way to perform a sanity check.
                // If feeding the same value multiple times needs to be supported,
                // the query should not be marked `no_hash`.
                bug!(
                    "Trying to feed an already recorded value for query {dep_kind:?} key={key:?}:\n\
                    old value: {old}\nnew value: {value}",
                    old = format_value(&old),
                    value = format_value(&value),
                )
            }
        }
        None => {
            // There is no cached value for this key, so feed the query by
            // adding the provided value to the cache.
            let dep_node = dep_graph::DepNode::construct(tcx, dep_kind, &key);
            let dep_node_index = tcx.dep_graph.with_feed_task(
                dep_node,
                tcx,
                &value,
                query_vtable.hash_result,
                query_vtable.format_value,
            );
            cache.complete(key, value, dep_node_index);
        }
    }
}
