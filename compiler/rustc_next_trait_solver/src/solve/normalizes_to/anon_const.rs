use rustc_type_ir::{self as ty, Interner, TypingMode};
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
        if self.typing_mode() == TypingMode::Coherence
            && self.cx().anon_const_kind(goal.predicate.alias.def_id) == ty::AnonConstKind::OGCA
        {
            // During coherence, OGCA consts should be normalized ambiguously
            // because they are opaque but eventually resolved to a real value.
            // We don't want two OGCAs that have the same value to be treated
            // as distinct for coherence purposes. (Just like opaque types.)
            //
            // We can't rely on evaluate_const below because that particular wrapper
            // treats too-generic consts as a successful evaluation.
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        } else if let Some(normalized_const) = self.evaluate_const(
            goal.param_env,
            ty::UnevaluatedConst::new(
                goal.predicate.alias.def_id.try_into().unwrap(),
                goal.predicate.alias.args,
            ),
        ) {
            self.instantiate_normalizes_to_term(goal, normalized_const.into());
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        }
    }
}
