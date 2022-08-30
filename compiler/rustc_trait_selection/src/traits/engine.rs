use std::cell::RefCell;

use super::TraitEngine;
use super::{ChalkFulfillmentContext, FulfillmentContext};
use crate::infer::InferCtxtExt;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::{InferCtxt, InferOk};
use rustc_infer::traits::{
    FulfillmentError, Obligation, ObligationCause, PredicateObligation, TraitEngineExt as _,
};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::ToPredicate;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

pub trait TraitEngineExt<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self>;
    fn new_in_snapshot(tcx: TyCtxt<'tcx>) -> Box<Self>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.unstable_opts.chalk {
            Box::new(ChalkFulfillmentContext::new())
        } else {
            Box::new(FulfillmentContext::new())
        }
    }

    fn new_in_snapshot(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.unstable_opts.chalk {
            Box::new(ChalkFulfillmentContext::new())
        } else {
            Box::new(FulfillmentContext::new_in_snapshot())
        }
    }
}

/// Used if you want to have pleasant experience when dealing
/// with obligations outside of hir or mir typeck.
pub struct ObligationCtxt<'a, 'tcx> {
    pub infcx: &'a InferCtxt<'a, 'tcx>,
    engine: RefCell<Box<dyn TraitEngine<'tcx>>>,
}

impl<'a, 'tcx> ObligationCtxt<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
        Self { infcx, engine: RefCell::new(<dyn TraitEngine<'_>>::new(infcx.tcx)) }
    }

    pub fn new_in_snapshot(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
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
        let trait_ref = ty::TraitRef { def_id, substs: tcx.mk_substs_trait(ty, &[]) };
        self.register_obligation(Obligation {
            cause,
            recursion_depth: 0,
            param_env,
            predicate: ty::Binder::dummy(trait_ref).without_const().to_predicate(tcx),
        });
    }

    pub fn normalize<T: TypeFoldable<'tcx>>(
        &self,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> T {
        let infer_ok = self.infcx.partially_normalize_associated_types_in(cause, param_env, value);
        self.register_infer_ok_obligations(infer_ok)
    }

    pub fn equate_types(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Result<(), TypeError<'tcx>> {
        match self.infcx.at(cause, param_env).eq(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_obligations(obligations);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn select_all_or_error(&self) -> Vec<FulfillmentError<'tcx>> {
        self.engine.borrow_mut().select_all_or_error(self.infcx)
    }

    pub fn assumed_wf_types(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        span: Span,
        def_id: LocalDefId,
    ) -> FxHashSet<Ty<'tcx>> {
        let tcx = self.infcx.tcx;
        let assumed_wf_types = tcx.assumed_wf_types(def_id);
        let mut implied_bounds = FxHashSet::default();
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        let cause = ObligationCause::misc(span, hir_id);
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
            let normalized = self.normalize(cause.clone(), param_env, ty);
            implied_bounds.insert(normalized);
        }
        implied_bounds
    }
}
