//! Computes a normalizes-to (projection) goal for inherent associated types,
//! `#![feature(lazy_type_alias)]` and `#![feature(type_alias_impl_trait)]`.
//!
//! Since a free alias is never ambiguous, this just computes the `type_of` of
//! the alias and registers the where-clauses of the type alias.

use rustc_type_ir::{self as ty, Interner, Unnormalized};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, GoalSource, QueryResult};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_free_alias(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let free_alias = goal.predicate.alias;

        // Check where clauses
        self.add_goals(
            GoalSource::Misc,
            cx.predicates_of(free_alias.def_id())
                .iter_instantiated(cx, free_alias.args)
                .map(Unnormalized::skip_norm_wip)
                .map(|pred| goal.with(cx, pred)),
        );

        let actual = match free_alias.kind(cx) {
            ty::AliasTermKind::FreeTy { def_id } => {
                cx.type_of(def_id).instantiate(cx, free_alias.args).skip_norm_wip().into()
            }
            ty::AliasTermKind::FreeConst { def_id } if cx.is_type_const(def_id) => {
                cx.const_of_item(def_id).instantiate(cx, free_alias.args).skip_norm_wip().into()
            }
            ty::AliasTermKind::FreeConst { .. } => {
                return self.evaluate_const_and_instantiate_normalizes_to_term(
                    goal,
                    free_alias.expect_ct(cx),
                );
            }
            kind => panic!("expected free alias, found {kind:?}"),
        };

        self.instantiate_normalizes_to_term(goal, actual);
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
