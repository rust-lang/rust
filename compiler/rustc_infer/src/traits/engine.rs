use crate::infer::InferCtxt;
use crate::traits::Obligation;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, ToPredicate, Ty, WithConstness};

use super::FulfillmentError;
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
        let trait_ref = ty::TraitRef { def_id, substs: infcx.tcx.mk_substs_trait(ty, &[]) };
        self.register_predicate_obligation(
            infcx,
            Obligation {
                cause,
                recursion_depth: 0,
                param_env,
                predicate: ty::Binder::dummy(trait_ref).without_const().to_predicate(infcx.tcx),
            },
        );
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

    fn select_all_with_constness_or_error(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        _constness: hir::Constness,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        self.select_all_or_error(infcx)
    }

    fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    // FIXME(fee1-dead) this should not provide a default body for chalk as chalk should be updated
    fn select_with_constness_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        _constness: hir::Constness,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        self.select_where_possible(infcx)
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>>;

    fn relationships(&mut self) -> &mut FxHashMap<ty::TyVid, ty::FoundRelationships>;
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
