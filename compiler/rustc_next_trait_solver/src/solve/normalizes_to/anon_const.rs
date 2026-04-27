use rustc_type_ir::{self as ty, Interner};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, QueryResult};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn normalize_anon_const(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let uv = goal.predicate.alias.expect_ct(self.cx());
        // keep legacy behavior for array repeat expressions:
        // when a constant is too generic to be evaluated, the legacy behavior is to return
        // Ambiguous, whereas evaluate_const_and_instantiate_normalizes_to_term structurally
        // instantiates to itself and returns Yes (if there are no inference variables)
        let is_repeat_expr =
            cx.anon_const_kind(uv.def.into()) == ty::AnonConstKind::RepeatExprCount;
        if is_repeat_expr {
            if let Some(normalized_const) = self.evaluate_const(goal.param_env, uv) {
                self.instantiate_normalizes_to_term(goal, normalized_const.into());
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            } else {
                self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
            }
        } else {
            self.evaluate_const_and_instantiate_normalizes_to_term(goal, uv)
        }
    }
}
