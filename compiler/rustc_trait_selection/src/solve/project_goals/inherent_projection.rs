//! Computes a normalizes-to (projection) goal for inherent associated types,
//! `#![feature(inherent_associated_type)]`. Since astconv already determines
//! which impl the IAT is being projected from, we just:
//! 1. instantiate substs,
//! 2. equate the self type, and
//! 3. instantiate and register where clauses.
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty;

use super::EvalCtxt;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn normalize_inherent_associated_type(
        &mut self,
        goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let inherent = goal.predicate.projection_ty;
        let expected = goal.predicate.term.ty().expect("inherent consts are treated separately");

        let impl_def_id = tcx.parent(inherent.def_id);
        let impl_substs = self.fresh_args_for_item(impl_def_id);

        // Equate impl header and add impl where clauses
        self.eq(
            goal.param_env,
            inherent.self_ty(),
            tcx.type_of(impl_def_id).instantiate(tcx, impl_substs),
        )?;

        // Equate IAT with the RHS of the project goal
        let inherent_substs = inherent.rebase_inherent_args_onto_impl(impl_substs, tcx);
        self.eq(
            goal.param_env,
            expected,
            tcx.type_of(inherent.def_id).instantiate(tcx, inherent_substs),
        )
        .expect("expected goal term to be fully unconstrained");

        // Check both where clauses on the impl and IAT
        self.add_goals(
            tcx.predicates_of(inherent.def_id)
                .instantiate(tcx, inherent_substs)
                .into_iter()
                .map(|(pred, _)| goal.with(tcx, pred)),
        );

        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
