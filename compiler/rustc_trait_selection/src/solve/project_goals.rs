use super::EvalCtxt;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty::{self, ProjectionPredicate};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        match goal.predicate.term.unpack() {
            ty::TermKind::Ty(term) => {
                let alias = goal.predicate.projection_ty.to_ty(self.tcx());
                self.eq(goal.param_env, alias, term)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            // FIXME(associated_const_equality): actually do something here.
            ty::TermKind::Const(_) => {
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }
}
