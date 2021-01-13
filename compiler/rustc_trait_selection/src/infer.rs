use crate::traits::query::outlives_bounds::InferCtxtExt as _;
use crate::traits::{self, TraitEngine, TraitEngineExt};

use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::traits::ObligationCause;
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::infer::canonical::{Canonical, CanonicalizedQueryResponse, QueryResponse};
use rustc_middle::traits::query::Fallible;
use rustc_middle::ty::{self, Ty, TypeFoldable};
use rustc_span::{Span, DUMMY_SP};

use std::fmt::Debug;

pub use rustc_infer::infer::*;

pub trait InferCtxtExt<'tcx> {
    fn type_is_copy_modulo_regions(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        span: Span,
    ) -> bool;

    fn partially_normalize_associated_types_in<T>(
        &self,
        span: Span,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>;
}

impl<'cx, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'cx, 'tcx> {
    fn type_is_copy_modulo_regions(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        span: Span,
    ) -> bool {
        let ty = self.resolve_vars_if_possible(ty);

        if !(param_env, ty).needs_infer() {
            return ty.is_copy_modulo_regions(self.tcx.at(span), param_env);
        }

        let copy_def_id = self.tcx.require_lang_item(LangItem::Copy, None);

        // This can get called from typeck (by euv), and `moves_by_default`
        // rightly refuses to work with inference variables, but
        // moves_by_default has a cache, which we want to use in other
        // cases.
        traits::type_known_to_meet_bound_modulo_regions(self, param_env, ty, copy_def_id, span)
    }

    /// Normalizes associated types in `value`, potentially returning
    /// new obligations that must further be processed.
    fn partially_normalize_associated_types_in<T>(
        &self,
        span: Span,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        debug!("partially_normalize_associated_types_in(value={:?})", value);
        let mut selcx = traits::SelectionContext::new(self);
        let cause = ObligationCause::misc(span, body_id);
        let traits::Normalized { value, obligations } =
            traits::normalize(&mut selcx, param_env, cause, value);
        debug!(
            "partially_normalize_associated_types_in: result={:?} predicates={:?}",
            value, obligations
        );
        InferOk { value, obligations }
    }
}

pub trait InferCtxtBuilderExt<'tcx> {
    fn enter_canonical_trait_query<K, R>(
        &mut self,
        canonical_key: &Canonical<'tcx, K>,
        operation: impl FnOnce(&InferCtxt<'_, 'tcx>, &mut dyn TraitEngine<'tcx>, K) -> Fallible<R>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, R>>
    where
        K: TypeFoldable<'tcx>,
        R: Debug + TypeFoldable<'tcx>,
        Canonical<'tcx, QueryResponse<'tcx, R>>: ArenaAllocatable<'tcx>;
}

impl<'tcx> InferCtxtBuilderExt<'tcx> for InferCtxtBuilder<'tcx> {
    /// The "main method" for a canonicalized trait query. Given the
    /// canonical key `canonical_key`, this method will create a new
    /// inference context, instantiate the key, and run your operation
    /// `op`. The operation should yield up a result (of type `R`) as
    /// well as a set of trait obligations that must be fully
    /// satisfied. These obligations will be processed and the
    /// canonical result created.
    ///
    /// Returns `NoSolution` in the event of any error.
    ///
    /// (It might be mildly nicer to implement this on `TyCtxt`, and
    /// not `InferCtxtBuilder`, but that is a bit tricky right now.
    /// In part because we would need a `for<'tcx>` sort of
    /// bound for the closure and in part because it is convenient to
    /// have `'tcx` be free on this function so that we can talk about
    /// `K: TypeFoldable<'tcx>`.)
    fn enter_canonical_trait_query<K, R>(
        &mut self,
        canonical_key: &Canonical<'tcx, K>,
        operation: impl FnOnce(&InferCtxt<'_, 'tcx>, &mut dyn TraitEngine<'tcx>, K) -> Fallible<R>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, R>>
    where
        K: TypeFoldable<'tcx>,
        R: Debug + TypeFoldable<'tcx>,
        Canonical<'tcx, QueryResponse<'tcx, R>>: ArenaAllocatable<'tcx>,
    {
        self.enter_with_canonical(
            DUMMY_SP,
            canonical_key,
            |ref infcx, key, canonical_inference_vars| {
                let mut fulfill_cx = TraitEngine::new(infcx.tcx);
                let value = operation(infcx, &mut *fulfill_cx, key)?;
                infcx.make_canonicalized_query_response(
                    canonical_inference_vars,
                    value,
                    &mut *fulfill_cx,
                )
            },
        )
    }
}

pub trait OutlivesEnvironmentExt<'tcx> {
    fn add_implied_bounds(
        &mut self,
        infcx: &InferCtxt<'a, 'tcx>,
        fn_sig_tys: &[Ty<'tcx>],
        body_id: hir::HirId,
        span: Span,
    );
}

impl<'tcx> OutlivesEnvironmentExt<'tcx> for OutlivesEnvironment<'tcx> {
    /// This method adds "implied bounds" into the outlives environment.
    /// Implied bounds are outlives relationships that we can deduce
    /// on the basis that certain types must be well-formed -- these are
    /// either the types that appear in the function signature or else
    /// the input types to an impl. For example, if you have a function
    /// like
    ///
    /// ```
    /// fn foo<'a, 'b, T>(x: &'a &'b [T]) { }
    /// ```
    ///
    /// we can assume in the caller's body that `'b: 'a` and that `T:
    /// 'b` (and hence, transitively, that `T: 'a`). This method would
    /// add those assumptions into the outlives-environment.
    ///
    /// Tests: `src/test/ui/regions/regions-free-region-ordering-*.rs`
    fn add_implied_bounds(
        &mut self,
        infcx: &InferCtxt<'a, 'tcx>,
        fn_sig_tys: &[Ty<'tcx>],
        body_id: hir::HirId,
        span: Span,
    ) {
        debug!("add_implied_bounds()");

        for &ty in fn_sig_tys {
            let ty = infcx.resolve_vars_if_possible(ty);
            debug!("add_implied_bounds: ty = {}", ty);
            let implied_bounds = infcx.implied_outlives_bounds(self.param_env, body_id, ty, span);
            self.add_outlives_bounds(Some(infcx), implied_bounds)
        }
    }
}
