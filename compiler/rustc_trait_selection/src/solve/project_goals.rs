use crate::solve::GoalSource;

use super::EvalCtxt;
use rustc_infer::infer::InferCtxt;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty::{self, ProjectionPredicate};

impl<'tcx> EvalCtxt<'_, InferCtxt<'tcx>> {
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let projection_term = goal.predicate.projection_term.to_term(tcx);
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
