use crate::FnCtxt;
use rustc_infer::traits::solve::Goal;
use rustc_infer::traits::{self, ObligationCause};
use rustc_middle::ty::{self, Ty};
use rustc_trait_selection::solve::inspect::ProofTreeInferCtxtExt;
use rustc_trait_selection::solve::inspect::{InspectConfig, InspectGoal, ProofTreeVisitor};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn obligations_for_self_ty(
        &self,
        self_ty: ty::TyVid,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        if self.next_trait_solver() {
            self.obligations_for_self_ty_next(self_ty)
        } else {
            let ty_var_root = self.root_var(self_ty);
            let mut obligations = self.fulfillment_cx.borrow().pending_obligations();
            trace!("pending_obligations = {:#?}", obligations);
            obligations
                .retain(|obligation| self.predicate_has_self_ty(obligation.predicate, ty_var_root));
            obligations
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn predicate_has_self_ty(
        &self,
        predicate: ty::Predicate<'tcx>,
        expected_vid: ty::TyVid,
    ) -> bool {
        match predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                self.self_type_matches_expected_vid(data.self_ty(), expected_vid)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                self.self_type_matches_expected_vid(data.projection_ty.self_ty(), expected_vid)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
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
            | ty::PredicateKind::Ambiguous => false,
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn self_type_matches_expected_vid(&self, self_ty: Ty<'tcx>, expected_vid: ty::TyVid) -> bool {
        let self_ty = self.shallow_resolve(self_ty);
        debug!(?self_ty);

        match *self_ty.kind() {
            ty::Infer(ty::TyVar(found_vid)) => expected_vid == self.root_var(found_vid),
            _ => false,
        }
    }

    pub(crate) fn obligations_for_self_ty_next(
        &self,
        self_ty: ty::TyVid,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        let obligations = self.fulfillment_cx.borrow().pending_obligations();
        let mut obligations_for_self_ty = vec![];
        for obligation in obligations {
            let mut visitor = NestedObligationsForSelfTy {
                fcx: self,
                ty_var_root: self.root_var(self_ty),
                obligations_for_self_ty: &mut obligations_for_self_ty,
                root_cause: &obligation.cause,
            };

            let goal = Goal::new(self.tcx, obligation.param_env, obligation.predicate);
            self.visit_proof_tree(goal, &mut visitor);
        }
        obligations_for_self_ty
    }
}

struct NestedObligationsForSelfTy<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    ty_var_root: ty::TyVid,
    root_cause: &'a ObligationCause<'tcx>,
    obligations_for_self_ty: &'a mut Vec<traits::PredicateObligation<'tcx>>,
}

impl<'a, 'tcx> ProofTreeVisitor<'tcx> for NestedObligationsForSelfTy<'a, 'tcx> {
    type Result = ();

    fn config(&self) -> InspectConfig {
        // Using an intentionally low depth to minimize the chance of future
        // breaking changes in case we adapt the approach later on. This also
        // avoids any hangs for exponentially growing proof trees.
        InspectConfig { max_depth: 3 }
    }

    fn visit_goal(&mut self, inspect_goal: &InspectGoal<'_, 'tcx>) {
        let tcx = self.fcx.tcx;
        let goal = inspect_goal.goal();
        if self.fcx.predicate_has_self_ty(goal.predicate, self.ty_var_root) {
            self.obligations_for_self_ty.push(traits::Obligation::new(
                tcx,
                self.root_cause.clone(),
                goal.param_env,
                goal.predicate,
            ));
        }

        if let Some(candidate) = inspect_goal.unique_applicable_candidate() {
            candidate.visit_nested_no_probe(self)
        }
    }
}
