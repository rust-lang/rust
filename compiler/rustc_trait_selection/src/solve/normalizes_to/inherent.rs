//! Computes a normalizes-to (projection) goal for inherent associated types,
//! `#![feature(inherent_associated_type)]`. Since astconv already determines
//! which impl the IAT is being projected from, we just:
//! 1. instantiate generic parameters,
//! 2. equate the self type, and
//! 3. instantiate and register where clauses.
use rustc_middle::traits::solve::{Certainty, Goal, GoalSource, QueryResult};
use rustc_middle::ty;

use crate::solve::EvalCtxt;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn normalize_inherent_associated_type(
        &mut self,
        goal: Goal<'tcx, ty::NormalizesTo<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let inherent = goal.predicate.alias;
        let expected = goal.predicate.term.ty().expect("inherent consts are treated separately");

        let impl_def_id = tcx.parent(inherent.def_id);
        let impl_args = self.fresh_args_for_item(impl_def_id);

        // Equate impl header and add impl where clauses
        self.eq(
            goal.param_env,
            inherent.self_ty(),
            tcx.type_of(impl_def_id).instantiate(tcx, impl_args),
        )?;

        // Equate IAT with the RHS of the project goal
        let inherent_args = inherent.rebase_inherent_args_onto_impl(impl_args, tcx);
        self.eq(
            goal.param_env,
            expected,
            tcx.type_of(inherent.def_id).instantiate(tcx, inherent_args),
        )
        .expect("expected goal term to be fully unconstrained");

        // Check both where clauses on the impl and IAT
        //
        // FIXME(-Znext-solver=coinductive): I think this should be split
        // and we tag the impl bounds with `GoalSource::ImplWhereBound`?
        // Right not this includes both the impl and the assoc item where bounds,
        // and I don't think the assoc item where-bounds are allowed to be coinductive.
        self.add_goals(
            GoalSource::Misc,
            tcx.predicates_of(inherent.def_id)
                .instantiate(tcx, inherent_args)
                .into_iter()
                .map(|(pred, _)| goal.with(tcx, pred)),
        );

        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
