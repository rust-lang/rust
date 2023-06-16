use super::EvalCtxt;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty::ProjectionPredicate;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let alias = goal.predicate.projection_ty.to_ty(self.tcx()).into();
        self.eq(goal.param_env, alias, goal.predicate.term)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
