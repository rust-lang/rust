//! Helper functions that serve as the immediate implementation of
//! `tcx.$query(..)` and its variations.

use std::ops::Deref;

use rustc_hir::def_id::LocalDefId;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::query::erase::{self, Erasable, Erased};
use crate::query::job::try_get_cached;
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
