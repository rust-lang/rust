use std::cell::RefCell;
use std::fmt::Debug;

use super::TraitEngine;
use super::{ChalkFulfillmentContext, FulfillmentContext};
use crate::solve::FulfillmentCtxt as NextFulfillmentCtxt;
use crate::traits::NormalizeExt;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::{
    Canonical, CanonicalQueryResponse, CanonicalVarValues, QueryResponse,
};
use rustc_infer::infer::{InferCtxt, InferOk};
use rustc_infer::traits::query::Fallible;
use rustc_infer::traits::{
    FulfillmentError, Obligation, ObligationCause, PredicateObligation, TraitEngineExt as _,
};
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::ToPredicate;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::TraitSolver;
use rustc_span::Span;

pub trait TraitEngineExt<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self>;
    fn new_in_snapshot(tcx: TyCtxt<'tcx>) -> Box<Self>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self> {
        match tcx.sess.opts.unstable_opts.trait_solver {
            TraitSolver::Classic => Box::new(FulfillmentContext::new()),
            TraitSolver::Chalk => Box::new(ChalkFulfillmentContext::new()),
            TraitSolver::Next => Box::new(NextFulfillmentCtxt::new()),
        }
    }

    fn new_in_snapshot(tcx: TyCtxt<'tcx>) -> Box<Self> {
        match tcx.sess.opts.unstable_opts.trait_solver {
            TraitSolver::Classic => Box::new(FulfillmentContext::new_in_snapshot()),
            TraitSolver::Chalk => Box::new(ChalkFulfillmentContext::new_in_snapshot()),
            TraitSolver::Next => Box::new(NextFulfillmentCtxt::new()),
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
        Self { infcx, engine: RefCell::new(<dyn TraitEngine<'_>>::new(infcx.tcx)) }
    }

    pub fn new_in_snapshot(infcx: &'a InferCtxt<'tcx>) -> Self {
        Self { infcx, engine: RefCell::new(<dyn TraitEngine<'_>>::new_in_snapshot(infcx.tcx)) }
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
        let trait_ref = tcx.mk_trait_ref(def_id, [ty]);
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
            .define_opaque_types(true)
            .eq_exp(a_is_expected, a, b)
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
            .define_opaque_types(true)
            .eq(expected, actual)
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
            .define_opaque_types(true)
            .sup(expected, actual)
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
            .define_opaque_types(true)
            .sup(expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    pub fn select_where_possible(&self) -> Vec<FulfillmentError<'tcx>> {
        self.engine.borrow_mut().select_where_possible(self.infcx)
    }

    pub fn select_all_or_error(&self) -> Vec<FulfillmentError<'tcx>> {
        self.engine.borrow_mut().select_all_or_error(self.infcx)
    }

    pub fn assumed_wf_types(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        span: Span,
        def_id: LocalDefId,
    ) -> FxIndexSet<Ty<'tcx>> {
        let tcx = self.infcx.tcx;
        let assumed_wf_types = tcx.assumed_wf_types(def_id);
        let mut implied_bounds = FxIndexSet::default();
        let cause = ObligationCause::misc(span, def_id);
        for ty in assumed_wf_types {
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
            let normalized = self.normalize(&cause, param_env, ty);
            implied_bounds.insert(normalized);
        }
        implied_bounds
    }

    pub fn make_canonicalized_query_response<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
    ) -> Fallible<CanonicalQueryResponse<'tcx, T>>
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
