//! Computes a normalizes-to (projection) goal for inherent associated types,
//! `#![feature(lazy_type_alias)]` and `#![feature(type_alias_impl_trait)]`.
//!
//! Since a weak alias is never ambiguous, this just computes the `type_of` of
//! the alias and registers the where-clauses of the type alias.

use rustc_type_ir::{self as ty, Interner};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, GoalSource, QueryResult};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_weak_type(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let weak_ty = goal.predicate.alias;

        // Check where clauses
        self.add_goals(
            GoalSource::Misc,
            cx.predicates_of(weak_ty.def_id)
                .iter_instantiated(cx, weak_ty.args)
                .map(|pred| goal.with(cx, pred)),
        );

        let actual = cx.type_of(weak_ty.def_id).instantiate(cx, weak_ty.args);
        self.instantiate_normalizes_to_term(goal, actual.into());

        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
