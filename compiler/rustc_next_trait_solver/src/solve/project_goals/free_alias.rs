//! Computes a projection goal for inherent associated types,
//! `#![feature(lazy_type_alias)]` and `#![feature(type_alias_impl_trait)]`.
//!
//! Since a free alias is never ambiguous, this just computes the `type_of` of
//! the alias and registers the where-clauses of the type alias.

use rustc_type_ir::solve::QueryResultOrRerunNonErased;
use rustc_type_ir::{self as ty, Interner, Unnormalized};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, GoalSource};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_free_alias(
        &mut self,
        goal: Goal<I, ty::ProjectionPredicate<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        let free_alias = goal.predicate.projection_term;

        // Check where clauses
        self.add_goals(
            GoalSource::Misc,
            cx.predicates_of(free_alias.def_id())
                .iter_instantiated(cx, free_alias.args)
                .map(Unnormalized::skip_norm_wip)
                .map(|pred| goal.with(cx, pred)),
        );

        let actual = match free_alias.kind {
            ty::AliasTermKind::FreeTy { def_id } => {
                cx.type_of(def_id.into()).instantiate(cx, free_alias.args).skip_norm_wip().into()
            }
            ty::AliasTermKind::FreeConst { def_id } if cx.is_type_const(def_id.into()) => cx
                .const_of_item(def_id.into())
                .instantiate(cx, free_alias.args)
                .skip_norm_wip()
                .into(),
            ty::AliasTermKind::FreeConst { .. } => {
                return self.evaluate_const_and_instantiate_projection_term(
                    goal.param_env,
                    free_alias,
                    goal.predicate.term,
                    free_alias.expect_ct(),
                );
            }
            kind => panic!("expected free alias, found {kind:?}"),
        };

        self.eq(goal.param_env, goal.predicate.term, actual)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
