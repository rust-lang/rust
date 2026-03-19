//! Helper functions that serve as the immediate implementation of
//! `tcx.$query(..)` and its variations.

use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::dep_graph;
use crate::dep_graph::DepNodeKey;
use crate::query::erase::{self, Erasable, Erased};
use crate::query::{EnsureMode, QueryCache, QueryMode, QueryVTable};
use crate::ty::TyCtxt;

/// Checks whether there is already a value for this key in the in-memory
/// query cache, returning that value if present.
///
/// (Also performs some associated bookkeeping, if a value was found.)
#[inline(always)]
fn try_get_cached<'tcx, C>(tcx: TyCtxt<'tcx>, cache: &C, key: C::Key) -> Option<C::Value>
where
    C: QueryCache,
{
    match cache.lookup(&key) {
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
pub(crate) fn query_get_at<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
) -> C::Value
where
    C: QueryCache,
{
    match try_get_cached(tcx, &query.cache, key) {
        Some(value) => value,
        None => (query.execute_query_fn)(tcx, span, key, QueryMode::Get).unwrap(),
    }
}

/// Shared implementation of `tcx.ensure_ok().$query(..)` and
/// `tcx.ensure_done().$query(..)` for all queries.
#[inline]
pub(crate) fn query_ensure_ok_or_done<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    ensure_mode: EnsureMode,
) where
    C: QueryCache,
{
    match try_get_cached(tcx, &query.cache, key) {
        Some(_value) => {}
        None => {
            (query.execute_query_fn)(tcx, DUMMY_SP, key, QueryMode::Ensure { ensure_mode });
        }
    }
}

/// Implementation of `tcx.ensure_result().$query(..)` for queries that
/// return `Result<_, ErrorGuaranteed>`.
#[inline]
pub(crate) fn query_ensure_result<'tcx, C, T>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
) -> Result<(), ErrorGuaranteed>
where
    C: QueryCache<Value = Erased<Result<T, ErrorGuaranteed>>>,
    Result<T, ErrorGuaranteed>: Erasable,
{
    match try_get_cached(tcx, &query.cache, key) {
        Some(value) => erase::restore_val(value).map(drop),
        None => {
            match (query.execute_query_fn)(
                tcx,
                DUMMY_SP,
                key,
                QueryMode::Ensure { ensure_mode: EnsureMode::Ok },
            ) {
                // We executed the query. If the restored result is `Ok(_)`, dropping it converts
                // it to `Ok(())`. If the restored result is `Err(guar)`, we pass that along.
                Some(res) => erase::restore_val(res).map(drop),

                // Reaching here means we didn't execute the query, but we can just assume the
                // query succeeded, because it was green in the incremental cache. If it is green,
                // that means that the previous compilation that wrote to the incremental cache
                // compiles successfully. That is only possible if the cache entry was `Ok(())`, so
                // we emit that here, without actually encoding the `Result` in the cache or
                // loading it from there.
                None => Ok(()),
            }
        }
    }
}

/// "Feeds" a feedable query by adding a given key/value pair to its in-memory cache.
/// Called by macro-generated methods of [`rustc_middle::ty::TyCtxtFeed`].
pub(crate) fn query_feed<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    value: C::Value,
) where
    C: QueryCache,
    C::Key: DepNodeKey<'tcx>,
{
    let format_value = query.format_value;

    // Check whether the in-memory cache already has a value for this key.
    match try_get_cached(tcx, &query.cache, key) {
        Some(old) => {
            // The query already has a cached value for this key.
            // That's OK if both values are the same, i.e. they have the same hash,
            // so now we check their hashes.
            if let Some(hash_value_fn) = query.hash_value_fn {
                let (old_hash, value_hash) = tcx.with_stable_hashing_context(|ref mut hcx| {
                    (hash_value_fn(hcx, &old), hash_value_fn(hcx, &value))
                });
                if old_hash != value_hash {
                    // We have an inconsistency. This can happen if one of the two
                    // results is tainted by errors. In this case, delay a bug to
                    // ensure compilation is doomed, and keep the `old` value.
                    tcx.dcx().delayed_bug(format!(
                        "Trying to feed an already recorded value for query {query:?} key={key:?}:\n\
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
                    "Trying to feed an already recorded value for query {query:?} key={key:?}:\n\
                    old value: {old}\nnew value: {value}",
                    old = format_value(&old),
                    value = format_value(&value),
                )
            }
        }
        None => {
            // There is no cached value for this key, so feed the query by
            // adding the provided value to the cache.
            let dep_node = dep_graph::DepNode::construct(tcx, query.dep_kind, &key);
            let dep_node_index = tcx.dep_graph.with_feed_task(
                dep_node,
                tcx,
                &value,
                query.hash_value_fn,
                query.format_value,
            );
            query.cache.complete(key, value, dep_node_index);
        }
    }
}
