//! Helper functions that serve as the immediate implementation of
//! `tcx.$query(..)` and its variations.

use rustc_data_structures::assert_matches;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::query::erase::{self, Erasable, Erased};
use crate::query::plumbing::QueryVTable;
use crate::query::{EnsureMode, QueryCache, QueryMode};
use crate::ty::TyCtxt;

/// Checks whether there is already a value for this key in the in-memory
/// query cache, returning that value if present.
///
/// (Also performs some associated bookkeeping, if a value was found.)
#[inline(always)]
pub fn try_get_cached<'tcx, C>(tcx: TyCtxt<'tcx>, cache: &C, key: &C::Key) -> Option<C::Value>
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
pub(crate) fn query_get_at<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
) -> C::Value
where
    C: QueryCache,
{
    match try_get_cached(tcx, &query.cache, &key) {
        Some(value) => value,
        None => (query.execute_query_fn)(tcx, span, key, QueryMode::Get).unwrap(),
    }
}

/// Shared implementation of `tcx.ensure_ok().$query(..)` for most queries,
/// and `tcx.ensure_done().$query(..)` for all queries.
#[inline]
pub(crate) fn query_ensure<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    ensure_mode: EnsureMode,
) where
    C: QueryCache,
{
    if try_get_cached(tcx, &query.cache, &key).is_none() {
        (query.execute_query_fn)(tcx, DUMMY_SP, key, QueryMode::Ensure { ensure_mode });
    }
}

/// Shared implementation of `tcx.ensure_ok().$query(..)` for queries that
/// have the `return_result_from_ensure_ok` modifier.
#[inline]
pub(crate) fn query_ensure_error_guaranteed<'tcx, C, T>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    // This arg is needed to match the signature of `query_ensure`,
    // but should always be `EnsureMode::Ok`.
    ensure_mode: EnsureMode,
) -> Result<(), ErrorGuaranteed>
where
    C: QueryCache<Value = Erased<Result<T, ErrorGuaranteed>>>,
    Result<T, ErrorGuaranteed>: Erasable,
{
    assert_matches!(ensure_mode, EnsureMode::Ok);

    if let Some(res) = try_get_cached(tcx, &query.cache, &key) {
        erase::restore_val(res).map(drop)
    } else {
        (query.execute_query_fn)(tcx, DUMMY_SP, key, QueryMode::Ensure { ensure_mode })
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
