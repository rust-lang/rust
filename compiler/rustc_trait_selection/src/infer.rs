use std::fmt::Debug;

use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
pub use rustc_infer::infer::*;
use rustc_macros::extension;
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::infer::canonical::{
    Canonical, CanonicalQueryInput, CanonicalQueryResponse, QueryResponse,
};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{self, GenericArg, Ty, TyCtxt, TypeFoldable, TypeVisitableExt, Upcast};
use rustc_span::DUMMY_SP;
use tracing::instrument;

use crate::infer::at::ToTrace;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::{self, Obligation, ObligationCause, ObligationCtxt, SelectionContext};

#[extension(pub trait InferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    fn can_eq<T: ToTrace<'tcx>>(&self, param_env: ty::ParamEnv<'tcx>, a: T, b: T) -> bool {
        self.probe(|_| {
            let ocx = ObligationCtxt::new(self);
            let Ok(()) = ocx.eq(&ObligationCause::dummy(), param_env, a, b) else {
                return false;
            };
            ocx.select_where_possible().is_empty()
        })
    }

    fn type_is_copy_modulo_regions(&self, param_env: ty::ParamEnv<'tcx>, ty: Ty<'tcx>) -> bool {
        let ty = self.resolve_vars_if_possible(ty);

        if !(param_env, ty).has_infer() {
            return ty.is_copy_modulo_regions(self.tcx, param_env);
        }

        let copy_def_id = self.tcx.require_lang_item(LangItem::Copy, None);

        // This can get called from typeck (by euv), and `moves_by_default`
        // rightly refuses to work with inference variables, but
        // moves_by_default has a cache, which we want to use in other
        // cases.
        traits::type_known_to_meet_bound_modulo_regions(self, param_env, ty, copy_def_id)
    }

    fn type_is_sized_modulo_regions(&self, param_env: ty::ParamEnv<'tcx>, ty: Ty<'tcx>) -> bool {
        let lang_item = self.tcx.require_lang_item(LangItem::Sized, None);
        traits::type_known_to_meet_bound_modulo_regions(self, param_env, ty, lang_item)
    }

    /// Check whether a `ty` implements given trait(trait_def_id) without side-effects.
    ///
    /// The inputs are:
    ///
    /// - the def-id of the trait
    /// - the type parameters of the trait, including the self-type
    /// - the parameter environment
    ///
    /// Invokes `evaluate_obligation`, so in the event that evaluating
    /// `Ty: Trait` causes overflow, EvaluatedToAmbigStackDependent will be returned.
    #[instrument(level = "debug", skip(self, params), ret)]
    fn type_implements_trait(
        &self,
        trait_def_id: DefId,
        params: impl IntoIterator<Item: Into<GenericArg<'tcx>>>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> traits::EvaluationResult {
        let trait_ref = ty::TraitRef::new(self.tcx, trait_def_id, params);

        let obligation = traits::Obligation {
            cause: traits::ObligationCause::dummy(),
            param_env,
            recursion_depth: 0,
            predicate: trait_ref.upcast(self.tcx),
        };
        self.evaluate_obligation(&obligation).unwrap_or(traits::EvaluationResult::EvaluatedToErr)
    }

    /// Returns `Some` if a type implements a trait shallowly, without side-effects,
    /// along with any errors that would have been reported upon further obligation
    /// processing.
    ///
    /// - If this returns `Some([])`, then the trait holds modulo regions.
    /// - If this returns `Some([errors..])`, then the trait has an impl for
    /// the self type, but some nested obligations do not hold.
    /// - If this returns `None`, no implementation that applies could be found.
    ///
    /// FIXME(-Znext-solver): Due to the recursive nature of the new solver,
    /// this will probably only ever return `Some([])` or `None`.
    fn type_implements_trait_shallow(
        &self,
        trait_def_id: DefId,
        ty: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<Vec<traits::FulfillmentError<'tcx>>> {
        self.probe(|_snapshot| {
            let mut selcx = SelectionContext::new(self);
            match selcx.select(&Obligation::new(
                self.tcx,
                ObligationCause::dummy(),
                param_env,
                ty::TraitRef::new(self.tcx, trait_def_id, [ty]),
            )) {
                Ok(Some(selection)) => {
                    let ocx = ObligationCtxt::new_with_diagnostics(self);
                    ocx.register_obligations(selection.nested_obligations());
                    Some(ocx.select_all_or_error())
                }
                Ok(None) | Err(_) => None,
            }
        })
    }
}

#[extension(pub trait InferCtxtBuilderExt<'tcx>)]
impl<'tcx> InferCtxtBuilder<'tcx> {
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
    /// `K: TypeFoldable<TyCtxt<'tcx>>`.)
    fn enter_canonical_trait_query<K, R>(
        self,
        canonical_key: &CanonicalQueryInput<'tcx, K>,
        operation: impl FnOnce(&ObligationCtxt<'_, 'tcx>, K) -> Result<R, NoSolution>,
    ) -> Result<CanonicalQueryResponse<'tcx, R>, NoSolution>
    where
        K: TypeFoldable<TyCtxt<'tcx>>,
        R: Debug + TypeFoldable<TyCtxt<'tcx>>,
        Canonical<'tcx, QueryResponse<'tcx, R>>: ArenaAllocatable<'tcx>,
    {
        let (infcx, key, canonical_inference_vars) =
            self.build_with_canonical(DUMMY_SP, canonical_key);
        let ocx = ObligationCtxt::new(&infcx);
        let value = operation(&ocx, key)?;
        ocx.make_canonicalized_query_response(canonical_inference_vars, value)
    }
}
