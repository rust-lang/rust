use rustc_type_ir::{self as ty, Interner};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::{EvalCtxt, Goal, QueryResult};

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
        let uv = goal.predicate.alias.expect_ct(self.cx());
        self.evaluate_const_and_instantiate_normalizes_to_term(goal, uv)
    }
}
