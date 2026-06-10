use rustc_type_ir::solve::QueryResultOrRerunNonErased;
use rustc_type_ir::{self as ty, Interner};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::{EvalCtxt, Goal};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn normalize_anon_const(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I, I::UnevaluatedConstId>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        let uv = ty::UnevaluatedConst::new(
            cx,
            ty::UnevaluatedConstKind::Anon { def_id: goal.predicate.alias.kind },
            goal.predicate.alias.args,
        );
        let alias = ty::AliasTerm::from(uv);
        let goal = goal.with(cx, ty::NormalizesTo { alias, term: goal.predicate.term });
        self.evaluate_const_and_instantiate_normalizes_to_term(goal, uv)
    }
}
