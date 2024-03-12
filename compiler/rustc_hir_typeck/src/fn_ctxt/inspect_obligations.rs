use crate::FnCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_infer::traits;
use rustc_data_structures::captures::Captures;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn obligations_for_self_ty<'b>(
        &'b self,
        self_ty: ty::TyVid,
    ) -> impl DoubleEndedIterator<Item = traits::PredicateObligation<'tcx>> + Captures<'tcx> + 'b
    {
        let ty_var_root = self.root_var(self_ty);
        trace!("pending_obligations = {:#?}", self.fulfillment_cx.borrow().pending_obligations());

        self.fulfillment_cx.borrow().pending_obligations().into_iter().filter_map(
            move |obligation| match &obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(data))
                    if self.self_type_matches_expected_vid(
                        data.projection_ty.self_ty(),
                        ty_var_root,
                    ) =>
                {
                    Some(obligation)
                }
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(data))
                    if self.self_type_matches_expected_vid(data.self_ty(), ty_var_root) =>
                {
                    Some(obligation)
                }

                ty::PredicateKind::Clause(ty::ClauseKind::Trait(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(..))
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::NormalizesTo(..)
                | ty::PredicateKind::AliasRelate(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::Ambiguous => None,
            },
        )
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn self_type_matches_expected_vid(&self, self_ty: Ty<'tcx>, expected_vid: ty::TyVid) -> bool {
        let self_ty = self.shallow_resolve(self_ty);
        debug!(?self_ty);

        match *self_ty.kind() {
            ty::Infer(ty::TyVar(found_vid)) => {
                let found_vid = self.root_var(found_vid);
                debug!("self_type_matches_expected_vid - found_vid={:?}", found_vid);
                expected_vid == found_vid
            }
            _ => false,
        }
    }
}