use std::cell::RefCell;
use std::fmt::Debug;

use super::FulfillmentContext;
use super::TraitEngine;
use crate::solve::FulfillmentCtxt as NextFulfillmentCtxt;
use crate::traits::error_reporting::TypeErrCtxtExt;
use crate::traits::NormalizeExt;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::{
    Canonical, CanonicalQueryResponse, CanonicalVarValues, QueryResponse,
};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk};
use rustc_infer::traits::{
    FulfillmentError, Obligation, ObligationCause, PredicateObligation, TraitEngineExt as _,
};
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::ToPredicate;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::TraitSolver;

pub trait TraitEngineExt<'tcx> {
    fn new(infcx: &InferCtxt<'tcx>) -> Box<Self>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> {
    fn new(infcx: &InferCtxt<'tcx>) -> Box<Self> {
        match (infcx.tcx.sess.opts.unstable_opts.trait_solver, infcx.next_trait_solver()) {
            (TraitSolver::Classic, false) | (TraitSolver::NextCoherence, false) => {
                Box::new(FulfillmentContext::new(infcx))
            }
            (TraitSolver::Next | TraitSolver::NextCoherence, true) => {
                Box::new(NextFulfillmentCtxt::new(infcx))
            }
            _ => bug!(
                "incompatible combination of -Ztrait-solver flag ({:?}) and InferCtxt::next_trait_solver ({:?})",
                infcx.tcx.sess.opts.unstable_opts.trait_solver,
                infcx.next_trait_solver()
            ),
        }
    }
}

/// Used if you want to have pleasant experience when dealing
/// with obligations outside of hir or mir typeck.
pub struct ObligationCtxt<'a, 'tcx> {
    pub infcx: &'a InferCtxt<'tcx>,
    engine: RefCell<Box<dyn TraitEngine<'tcx>>>,
}

impl<'a, 'tcx> ObligationCtxt<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        Self { infcx, engine: RefCell::new(<dyn TraitEngine<'_>>::new(infcx)) }
    }

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
            predicate: ty::Binder::dummy(trait_ref).without_const().to_predicate(tcx),
        });
    }

    pub fn normalize<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> T {
        let infer_ok = self.infcx.at(&cause, param_env).normalize(value);
        self.register_infer_ok_obligations(infer_ok)
    }

    /// Makes `expected <: actual`.
    pub fn eq_exp<T>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        a_is_expected: bool,
        a: T,
        b: T,
    ) -> Result<(), TypeError<'tcx>>
    where
        T: ToTrace<'tcx>,
    {
        self.infcx
            .at(cause, param_env)
            .eq_exp(DefineOpaqueTypes::Yes, a_is_expected, a, b)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
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

    #[must_use]
    pub fn select_where_possible(&self) -> Vec<FulfillmentError<'tcx>> {
        self.engine.borrow_mut().select_where_possible(self.infcx)
    }

    #[must_use]
    pub fn select_all_or_error(&self) -> Vec<FulfillmentError<'tcx>> {
        self.engine.borrow_mut().select_all_or_error(self.infcx)
    }

    /// Resolves regions and reports errors.
    ///
    /// Takes ownership of the context as doing trait solving afterwards
    /// will result in region constraints getting ignored.
    pub fn resolve_regions_and_report_errors(
        self,
        generic_param_scope: LocalDefId,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        let errors = self.infcx.resolve_regions(&outlives_env);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(self.infcx.err_ctxt().report_region_errors(generic_param_scope, &errors))
        }
    }

    pub fn assumed_wf_types_and_report_errors(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        def_id: LocalDefId,
    ) -> Result<FxIndexSet<Ty<'tcx>>, ErrorGuaranteed> {
        self.assumed_wf_types(param_env, def_id)
            .map_err(|errors| self.infcx.err_ctxt().report_fulfillment_errors(&errors))
    }

    pub fn assumed_wf_types(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        def_id: LocalDefId,
    ) -> Result<FxIndexSet<Ty<'tcx>>, Vec<FulfillmentError<'tcx>>> {
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
