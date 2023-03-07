use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty;

use super::EvalCtxt;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn normalize_weak_type(
        &mut self,
        goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let weak_ty = goal.predicate.projection_ty;
        let expected = goal.predicate.term.ty().expect("no such thing as a const alias");

        let actual = tcx.type_of(weak_ty.def_id).subst(tcx, weak_ty.substs);
        self.eq(goal.param_env, expected, actual)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
