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
        if let Some(normalized_const) = self.evaluate_const(
            goal.param_env,
            ty::UnevaluatedConst::new(goal.predicate.alias.def_id, goal.predicate.alias.args),
        ) {
            self.instantiate_normalizes_to_term(goal, normalized_const.into());
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            // FIXME(BoxyUwU) FIXME(min_generic_const_args): I could not figure out how to write a test for this
            // as we don't currently support constants in the type system with impossible predicates. It may become
            // possible once `min_generic_const_args` has progressed more.

            // In coherence we should never consider an unevaluatable constant to be rigid. It may be failing due to
            // impossible predicates (cc #139000 #137972), or a `panic!`, either way we don't want this to influence
            // what impls are considered to overlap.
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        }
    }
}
