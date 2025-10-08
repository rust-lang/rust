//! Helper functions that serve as the immediate implementation of
//! `tcx.$query(..)` and its variations.

use rustc_query_system::query::{QueryCache, QueryMode, try_get_cached};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::query::IntoQueryParam;
use crate::query::erase::{self, Erase, EraseType};
use crate::ty::TyCtxt;

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
    let key = key.into_query_param();
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
    let key = key.into_query_param();
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
    Cache: QueryCache<Value = Erase<Result<T, ErrorGuaranteed>>>,
    Result<T, ErrorGuaranteed>: EraseType,
{
    let key = key.into_query_param();
    if let Some(res) = try_get_cached(tcx, query_cache, &key) {
        erase::restore(res).map(drop)
    } else {
        execute_query(tcx, DUMMY_SP, key, QueryMode::Ensure { check_cache })
            .map(erase::restore)
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
