use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::{self, TraitEngine, TraitEngineExt};

use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_infer::traits::ObligationCause;
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::infer::canonical::{Canonical, CanonicalizedQueryResponse, QueryResponse};
use rustc_middle::traits::query::Fallible;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::ToPredicate;
use rustc_middle::ty::WithConstness;
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
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>;

    /// Check whether a `ty` implements given trait(trait_def_id).
    /// The inputs are:
    ///
    /// - the def-id of the trait
    /// - the self type
    /// - the *other* type parameters of the trait, excluding the self-type
    /// - the parameter environment
    ///
    /// Invokes `evaluate_obligation`, so in the event that evaluating
    /// `Ty: Trait` causes overflow, EvaluatedToRecur (or EvaluatedToUnknown)
    /// will be returned.
    fn type_implements_trait(
        &self,
        trait_def_id: DefId,
        ty: Ty<'tcx>,
        params: SubstsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> traits::EvaluationResult;
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
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        debug!("partially_normalize_associated_types_in(value={:?})", value);
        let mut selcx = traits::SelectionContext::new(self);
        let traits::Normalized { value, obligations } =
            traits::normalize(&mut selcx, param_env, cause, value);
        debug!(
            "partially_normalize_associated_types_in: result={:?} predicates={:?}",
            value, obligations
        );
        InferOk { value, obligations }
    }

    fn type_implements_trait(
        &self,
        trait_def_id: DefId,
        ty: Ty<'tcx>,
        params: SubstsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> traits::EvaluationResult {
        debug!(
            "type_implements_trait: trait_def_id={:?}, type={:?}, params={:?}, param_env={:?}",
            trait_def_id, ty, params, param_env
        );

        let trait_ref =
            ty::TraitRef { def_id: trait_def_id, substs: self.tcx.mk_substs_trait(ty, params) };

        let obligation = traits::Obligation {
            cause: traits::ObligationCause::dummy(),
            param_env,
            recursion_depth: 0,
            predicate: ty::Binder::dummy(trait_ref).without_const().to_predicate(self.tcx),
        };
        self.evaluate_obligation(&obligation).unwrap_or(traits::EvaluationResult::EvaluatedToErr)
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
                let mut fulfill_cx = <dyn TraitEngine<'_>>::new(infcx.tcx);
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
