use rustc_type_ir::{self as ty, Interner, ProjectionPredicate};
use tracing::instrument;

use crate::infcx::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, GoalSource, QueryResult};

impl<Infcx, I> EvalCtxt<'_, Infcx>
where
    Infcx: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<I, ProjectionPredicate<I>>,
    ) -> QueryResult<I> {
        let tcx = self.interner();
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
