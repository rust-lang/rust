use std::cell::RefCell;
use std::fmt::Debug;

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::{
    Canonical, CanonicalQueryResponse, CanonicalVarValues, QueryResponse,
};
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk, RegionResolutionError, TypeTrace};
use rustc_infer::traits::PredicateObligations;
use rustc_macros::extension;
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::Relate;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, Upcast, Variance};

use super::{FromSolverError, FulfillmentContext, ScrubbedTraitError, TraitEngine};
use crate::error_reporting::InferCtxtErrorExt;
use crate::regions::InferCtxtRegionExt;
use crate::solve::{FulfillmentCtxt as NextFulfillmentCtxt, NextSolverError};
use crate::traits::fulfill::OldSolverError;
use crate::traits::{
    FulfillmentError, NormalizeExt, Obligation, ObligationCause, PredicateObligation,
    StructurallyNormalizeExt,
};

#[extension(pub trait TraitEngineExt<'tcx, E>)]
impl<'tcx, E> dyn TraitEngine<'tcx, E>
where
    E: FromSolverError<'tcx, NextSolverError<'tcx>> + FromSolverError<'tcx, OldSolverError<'tcx>>,
{
    fn new(infcx: &InferCtxt<'tcx>) -> Box<Self> {
        if infcx.next_trait_solver() {
            Box::new(NextFulfillmentCtxt::new(infcx))
        } else {
            assert!(
                !infcx.tcx.next_trait_solver_globally(),
                "using old solver even though new solver is enabled globally"
            );
            Box::new(FulfillmentContext::new(infcx))
        }
    }
}

/// Used if you want to have pleasant experience when dealing
/// with obligations outside of hir or mir typeck.
pub struct ObligationCtxt<'a, 'tcx, E = ScrubbedTraitError<'tcx>> {
    pub infcx: &'a InferCtxt<'tcx>,
    engine: RefCell<Box<dyn TraitEngine<'tcx, E>>>,
}

impl<'a, 'tcx> ObligationCtxt<'a, 'tcx, FulfillmentError<'tcx>> {
    pub fn new_with_diagnostics(infcx: &'a InferCtxt<'tcx>) -> Self {
        Self { infcx, engine: RefCell::new(<dyn TraitEngine<'tcx, _>>::new(infcx)) }
    }
}

impl<'a, 'tcx> ObligationCtxt<'a, 'tcx, ScrubbedTraitError<'tcx>> {
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        Self { infcx, engine: RefCell::new(<dyn TraitEngine<'tcx, _>>::new(infcx)) }
    }
}

impl<'a, 'tcx, E> ObligationCtxt<'a, 'tcx, E>
where
    E: 'tcx,
{
    pub fn register_obligation(&self, obligation: PredicateObligation<'tcx>) {
        self.engine.borrow_mut().register_predicate_obligation(self.infcx, obligation);
    }

    pub fn register_obligations(
        &self,
        obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
    ) {
        // Can't use `register_predicate_obligations` because the iterator
        // may also use this `ObligationCtxt`.
        for obligation in obligations {
            self.engine.borrow_mut().register_predicate_obligation(self.infcx, obligation)
        }
    }

    pub fn register_infer_ok_obligations<T>(&self, infer_ok: InferOk<'tcx, T>) -> T {
        let InferOk { value, obligations } = infer_ok;
        self.engine.borrow_mut().register_predicate_obligations(self.infcx, obligations);
        value
    }

    /// Requires that `ty` must implement the trait with `def_id` in
    /// the given environment. This trait must not have any type
    /// parameters (except for `Self`).
    pub fn register_bound(
        &self,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        def_id: DefId,
    ) {
        let tcx = self.infcx.tcx;
        let trait_ref = ty::TraitRef::new(tcx, def_id, [ty]);
        self.register_obligation(Obligation {
            cause,
            recursion_depth: 0,
            param_env,
            predicate: trait_ref.upcast(tcx),
        });
    }

    pub fn normalize<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> T {
        let infer_ok = self.infcx.at(cause, param_env).normalize(value);
        self.register_infer_ok_obligations(infer_ok)
    }

    pub fn eq<T: ToTrace<'tcx>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'tcx>> {
        self.infcx
            .at(cause, param_env)
            .eq(DefineOpaqueTypes::Yes, expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    pub fn eq_trace<T: Relate<TyCtxt<'tcx>>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        trace: TypeTrace<'tcx>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'tcx>> {
        self.infcx
            .at(cause, param_env)
            .eq_trace(DefineOpaqueTypes::Yes, trace, expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    /// Checks whether `expected` is a subtype of `actual`: `expected <: actual`.
    pub fn sub<T: ToTrace<'tcx>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'tcx>> {
        self.infcx
            .at(cause, param_env)
            .sub(DefineOpaqueTypes::Yes, expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    pub fn relate<T: ToTrace<'tcx>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        variance: Variance,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'tcx>> {
        self.infcx
            .at(cause, param_env)
            .relate(DefineOpaqueTypes::Yes, expected, variance, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    /// Checks whether `expected` is a supertype of `actual`: `expected :> actual`.
    pub fn sup<T: ToTrace<'tcx>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'tcx>> {
        self.infcx
            .at(cause, param_env)
            .sup(DefineOpaqueTypes::Yes, expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    /// Computes the least-upper-bound, or mutual supertype, of two values.
    pub fn lub<T: ToTrace<'tcx>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: T,
        actual: T,
    ) -> Result<T, TypeError<'tcx>> {
        self.infcx
            .at(cause, param_env)
            .lub(expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    #[must_use]
    pub fn select_where_possible(&self) -> Vec<E> {
        self.engine.borrow_mut().select_where_possible(self.infcx)
    }

    #[must_use]
    pub fn select_all_or_error(&self) -> Vec<E> {
        self.engine.borrow_mut().select_all_or_error(self.infcx)
    }

    /// Returns the not-yet-processed and stalled obligations from the
    /// `ObligationCtxt`.
    ///
    /// Takes ownership of the context as doing operations such as
    /// [`ObligationCtxt::eq`] afterwards will result in other obligations
    /// getting ignored. You can make a new `ObligationCtxt` if this
    /// needs to be done in a loop, for example.
    #[must_use]
    pub fn into_pending_obligations(self) -> PredicateObligations<'tcx> {
        self.engine.borrow().pending_obligations()
    }

    /// Resolves regions and reports errors.
    ///
    /// Takes ownership of the context as doing trait solving afterwards
    /// will result in region constraints getting ignored.
    pub fn resolve_regions_and_report_errors(
        self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Result<(), ErrorGuaranteed> {
        let errors = self.infcx.resolve_regions(body_id, param_env, assumed_wf_tys);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(self.infcx.err_ctxt().report_region_errors(body_id, &errors))
        }
    }

    /// Resolves regions and reports errors.
    ///
    /// Takes ownership of the context as doing trait solving afterwards
    /// will result in region constraints getting ignored.
    #[must_use]
    pub fn resolve_regions(
        self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.infcx.resolve_regions(body_id, param_env, assumed_wf_tys)
    }
}

impl<'tcx> ObligationCtxt<'_, 'tcx, FulfillmentError<'tcx>> {
    pub fn assumed_wf_types_and_report_errors(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        def_id: LocalDefId,
    ) -> Result<FxIndexSet<Ty<'tcx>>, ErrorGuaranteed> {
        self.assumed_wf_types(param_env, def_id)
            .map_err(|errors| self.infcx.err_ctxt().report_fulfillment_errors(errors))
    }
}

impl<'tcx> ObligationCtxt<'_, 'tcx, ScrubbedTraitError<'tcx>> {
    pub fn make_canonicalized_query_response<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
    ) -> Result<CanonicalQueryResponse<'tcx, T>, NoSolution>
    where
        T: Debug + TypeFoldable<TyCtxt<'tcx>>,
        Canonical<'tcx, QueryResponse<'tcx, T>>: ArenaAllocatable<'tcx>,
    {
        self.infcx.make_canonicalized_query_response(
            inference_vars,
            answer,
            &mut **self.engine.borrow_mut(),
        )
    }
}

impl<'tcx, E> ObligationCtxt<'_, 'tcx, E>
where
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    pub fn assumed_wf_types(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        def_id: LocalDefId,
    ) -> Result<FxIndexSet<Ty<'tcx>>, Vec<E>> {
        let tcx = self.infcx.tcx;
        let mut implied_bounds = FxIndexSet::default();
        let mut errors = Vec::new();
        for &(ty, span) in tcx.assumed_wf_types(def_id) {
            // FIXME(@lcnr): rustc currently does not check wf for types
            // pre-normalization, meaning that implied bounds are sometimes
            // incorrect. See #100910 for more details.
            //
            // Not adding the unnormalized types here mostly fixes that, except
            // that there are projections which are still ambiguous in the item definition
            // but do normalize successfully when using the item, see #98543.
            //
            // Anyways, I will hopefully soon change implied bounds to make all of this
            // sound and then uncomment this line again.

            // implied_bounds.insert(ty);
            let cause = ObligationCause::misc(span, def_id);
            match self
                .infcx
                .at(&cause, param_env)
                .deeply_normalize(ty, &mut **self.engine.borrow_mut())
            {
                // Insert well-formed types, ignoring duplicates.
                Ok(normalized) => drop(implied_bounds.insert(normalized)),
                Err(normalization_errors) => errors.extend(normalization_errors),
            };
        }

        if errors.is_empty() { Ok(implied_bounds) } else { Err(errors) }
    }

    pub fn deeply_normalize<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> Result<T, Vec<E>> {
        self.infcx.at(cause, param_env).deeply_normalize(value, &mut **self.engine.borrow_mut())
    }

    pub fn structurally_normalize_ty(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: Ty<'tcx>,
    ) -> Result<Ty<'tcx>, Vec<E>> {
        self.infcx
            .at(cause, param_env)
            .structurally_normalize_ty(value, &mut **self.engine.borrow_mut())
    }

    pub fn structurally_normalize_const(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: ty::Const<'tcx>,
    ) -> Result<ty::Const<'tcx>, Vec<E>> {
        self.infcx
            .at(cause, param_env)
            .structurally_normalize_const(value, &mut **self.engine.borrow_mut())
    }

    pub fn structurally_normalize_term(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: ty::Term<'tcx>,
    ) -> Result<ty::Term<'tcx>, Vec<E>> {
        self.infcx
            .at(cause, param_env)
            .structurally_normalize_term(value, &mut **self.engine.borrow_mut())
    }
}
