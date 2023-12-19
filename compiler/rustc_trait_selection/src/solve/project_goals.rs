use crate::solve::GoalSource;

use super::EvalCtxt;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty::{self, ProjectionPredicate};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let projection_term = match goal.predicate.term.unpack() {
            ty::TermKind::Ty(_) => goal.predicate.projection_ty.to_ty(tcx).into(),
            ty::TermKind::Const(_) => ty::Const::new_unevaluated(
                tcx,
                ty::UnevaluatedConst::new(
                    goal.predicate.projection_ty.def_id,
                    goal.predicate.projection_ty.args,
                ),
                tcx.type_of(goal.predicate.projection_ty.def_id)
                    .instantiate(tcx, goal.predicate.projection_ty.args),
            )
            .into(),
        };
        let goal = goal.with(
            tcx,
            ty::PredicateKind::AliasRelate(
                projection_term,
                goal.predicate.term,
                ty::AliasRelationDirection::Equate,
            ),
        );
        self.add_goal(GoalSource::Misc, goal);
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
