use crate::infer::InferCtxt;
use crate::ty::{self, Ty, TyCtxt, ToPredicate};
use crate::traits::Obligation;
use crate::hir::def_id::DefId;

use super::{ChalkFulfillmentContext, FulfillmentContext, FulfillmentError};
use super::{ObligationCause, PredicateObligation};

pub trait TraitEngine<'tcx>: 'tcx {
    fn normalize_projection_type(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        cause: ObligationCause<'tcx>,
    ) -> Ty<'tcx>;

    /// Requires that `ty` must implement the trait with `def_id` in
    /// the given environment. This trait must not have any type
    /// parameters (except for `Self`).
    fn register_bound(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        def_id: DefId,
        cause: ObligationCause<'tcx>,
    ) {
        let trait_ref = ty::TraitRef {
            def_id,
            substs: infcx.tcx.mk_substs_trait(ty, &[]),
        };
        self.register_predicate_obligation(infcx, Obligation {
            cause,
            recursion_depth: 0,
            param_env,
            predicate: trait_ref.to_predicate()
        });
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    );

    fn select_all_or_error(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>>;

    /// Retrieves the list of delayed generator witness predicates
    /// stored by this `TraitEngine`. This `TraitEngine` must have been
    /// created by `TraitEngine::with_delayed_generator_witness` - otherwise,
    /// this method will panic.
    ///
    /// Calling this method consumes the underling `Vec` - subsequent calls
    /// will return `None`.
    ///
    /// This method *MUST* be called on a `TraitEngine` created by
    /// `with_delayed_generator_witness` - if the `TraitEngine` is dropped
    /// without this method being called, a panic will occur. This ensures
    /// that the caller explicitly acknowledges these stored predicates -
    /// failure to do so will result in unsound code being accepted by
    /// the compiler.
    fn delayed_generator_obligations(&mut self) -> Option<Vec<PredicateObligation<'tcx>>>;
}

pub trait TraitEngineExt<'tcx> {
    fn register_predicate_obligations(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
    );
}

impl<T: ?Sized + TraitEngine<'tcx>> TraitEngineExt<'tcx> for T {
    fn register_predicate_obligations(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
    ) {
        for obligation in obligations {
            self.register_predicate_obligation(infcx, obligation);
        }
    }
}

impl dyn TraitEngine<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.debugging_opts.chalk {
            Box::new(ChalkFulfillmentContext::new())
        } else {
            Box::new(FulfillmentContext::new())
        }
    }

    /// Creates a `TraitEngine` in a special 'delay generator witness' mode.
    /// This imposes additional requirements for the caller in order to avoid
    /// accepting unsound code, and should only be used by `FnCtxt`. All other
    /// users of `TraitEngine` should use `TraitEngine::new`
    ///
    /// A `TraitEngine` returned by this constructor will not attempt
    /// to resolve any `GeneratorWitness` predicates involving auto traits,
    /// Specifically, predicates of the form:
    ///
    /// `<GeneratorWitness>: MyTrait` where `MyTrait` is an auto-trait
    /// will be stored for later retrieval by `delayed_generator_obligations`.
    /// The caller of this code *MUST* register these predicates with a
    /// regular `TraitEngine` (created with `TraitEngine::new`) at some point.
    /// Otherwise, these predicates will never be evaluated, resulting in
    /// unsound programs being accepted by the compiler.
    pub fn with_delayed_generator_witness(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.debugging_opts.chalk {
            Box::new(ChalkFulfillmentContext::with_delayed_generator_witness())
        } else {
            Box::new(FulfillmentContext::with_delayed_generator_witness())
        }

    }
}
