//! Helper functions that serve as the immediate implementation of
//! `tcx.$query(..)` and its variations.

use std::ops::Deref;

use rustc_hir::def_id::LocalDefId;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::dep_graph;
use crate::dep_graph::DepNodeKey;
use crate::query::erase::{self, Erasable, Erased};
use crate::query::{IntoQueryKey, QueryCache, QueryMode, QueryVTable};
use crate::ty::{self, TyCtxt};

#[derive(Copy, Clone)]
pub struct TyCtxtAt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub span: Span,
}

impl<'tcx> Deref for TyCtxtAt<'tcx> {
    type Target = TyCtxt<'tcx>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.tcx
    }
}

#[derive(Copy, Clone)]
#[must_use]
pub struct TyCtxtEnsureOk<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

#[derive(Copy, Clone)]
#[must_use]
pub struct TyCtxtEnsureResult<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

#[derive(Copy, Clone)]
#[must_use]
pub struct TyCtxtEnsureDone<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

impl<'tcx> TyCtxtEnsureOk<'tcx> {
    pub fn typeck(self, def_id: impl IntoQueryKey<LocalDefId>) {
        self.typeck_root(
            self.tcx.typeck_root_def_id(def_id.into_query_key().to_def_id()).expect_local(),
        )
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn typeck(self, def_id: impl IntoQueryKey<LocalDefId>) -> &'tcx ty::TypeckResults<'tcx> {
        self.typeck_root(
            self.typeck_root_def_id(def_id.into_query_key().to_def_id()).expect_local(),
        )
    }

    /// Returns a transparent wrapper for `TyCtxt` which uses
    /// `span` as the location of queries performed through it.
    #[inline(always)]
    pub fn at(self, span: Span) -> TyCtxtAt<'tcx> {
        TyCtxtAt { tcx: self, span }
    }

    /// FIXME: `ensure_ok`'s effects are subtle. Is this comment fully accurate?
    ///
    /// Wrapper that calls queries in a special "ensure OK" mode, for callers
    /// that don't need the return value and just want to invoke a query for
    /// its potential side-effect of emitting fatal errors.
    ///
    /// This can be more efficient than a normal query call, because if the
    /// query's inputs are all green, the call can return immediately without
    /// needing to obtain a value (by decoding one from disk or by executing
    /// the query).
    ///
    /// (As with all query calls, execution is also skipped if the query result
    /// is already cached in memory.)
    ///
    /// ## WARNING
    /// A subsequent normal call to the same query might still cause it to be
    /// executed! This can occur when the inputs are all green, but the query's
    /// result is not cached on disk, so the query must be executed to obtain a
    /// return value.
    ///
    /// Therefore, this call mode is not appropriate for callers that want to
    /// ensure that the query is _never_ executed in the future.
    #[inline(always)]
    pub fn ensure_ok(self) -> TyCtxtEnsureOk<'tcx> {
        TyCtxtEnsureOk { tcx: self }
    }

    /// This is a variant of `ensure_ok` only usable with queries that return
    /// `Result<_, ErrorGuaranteed>`. Queries calls through this function will
    /// return `Result<(), ErrorGuaranteed>`. I.e. the error status is returned
    /// but nothing else. As with `ensure_ok`, this can be more efficient than
    /// a normal query call.
    #[inline(always)]
    pub fn ensure_result(self) -> TyCtxtEnsureResult<'tcx> {
        TyCtxtEnsureResult { tcx: self }
    }

    /// Wrapper that calls queries where callers don't need the return value and
    /// just want to guarantee that the query won't be executed in the future.
    ///
    /// This is useful for queries that read from a [`Steal`] value, to ensure
    /// that they are executed before the query that will steal the value.
    ///
    /// Currently this causes the query to be executed normally, but this behavior may change.
    ///
    /// [`Steal`]: rustc_data_structures::steal::Steal
    #[inline(always)]
    pub fn ensure_done(self) -> TyCtxtEnsureDone<'tcx> {
        TyCtxtEnsureDone { tcx: self }
    }
}

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

/// Implementation of `tcx.ensure_ok().$query(..)` for all queries.
#[inline]
pub(crate) fn query_ensure_ok<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
) where
    C: QueryCache,
{
    match try_get_cached(tcx, &query.cache, key) {
        Some(_value) => {}
        None => {
            (query.execute_query_fn)(tcx, DUMMY_SP, key, QueryMode::EnsureOk);
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
    let convert = |value: Erased<Result<T, ErrorGuaranteed>>| -> Result<(), ErrorGuaranteed> {
        match erase::restore_val(value) {
            Ok(_) => Ok(()),
            Err(guar) => Err(guar),
        }
    };

    match try_get_cached(tcx, &query.cache, key) {
        Some(value) => convert(value),
        None => {
            match (query.execute_query_fn)(tcx, DUMMY_SP, key, QueryMode::EnsureOk) {
                // We executed the query. Convert the successful result.
                Some(res) => convert(res),

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
