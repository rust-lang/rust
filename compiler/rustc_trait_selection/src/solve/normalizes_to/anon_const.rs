use crate::solve::EvalCtxt;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn normalize_anon_const(
        &mut self,
        goal: Goal<'tcx, ty::NormalizesTo<'tcx>>,
    ) -> QueryResult<'tcx> {
        if let Some(normalized_const) = self.try_const_eval_resolve(
            goal.param_env,
            ty::UnevaluatedConst::new(goal.predicate.alias.def_id, goal.predicate.alias.args),
            self.tcx()
                .type_of(goal.predicate.alias.def_id)
                .no_bound_vars()
                .expect("const ty should not rely on other generics"),
        ) {
            self.eq(goal.param_env, normalized_const, goal.predicate.term.ct().unwrap())?;
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        }
    }
}
