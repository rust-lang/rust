//! Computes a normalizes-to (projection) goal for inherent associated types,
//! `#![feature(lazy_type_alias)]` and `#![feature(type_alias_impl_trait)]`.
//!
//! Since a weak alias is not ambiguous, this just computes the `type_of` of
//! the alias and registers the where-clauses of the type alias.
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

        let actual = tcx.type_of(weak_ty.def_id).instantiate(tcx, weak_ty.args);
        self.eq(goal.param_env, expected, actual)?;

        // Check where clauses
        self.add_goals(
            tcx.predicates_of(weak_ty.def_id)
                .instantiate(tcx, weak_ty.args)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred)),
        );

        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
