//! Computes a normalizes-to (projection) goal for inherent associated types,
//! `#![feature(inherent_associated_type)]`. Since HIR ty lowering already determines
//! which impl the IAT is being projected from, we just:
//! 1. instantiate generic parameters,
//! 2. equate the self type, and
//! 3. instantiate and register where clauses.

use rustc_type_ir::{self as ty, Interner};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, GoalSource, QueryResult};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_inherent_associated_term(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let inherent = goal.predicate.alias;

        let impl_def_id = cx.parent(inherent.def_id);
        let impl_args = self.fresh_args_for_item(impl_def_id);

        // Equate impl header and add impl where clauses
        self.eq(
            goal.param_env,
            inherent.self_ty(),
            cx.type_of(impl_def_id).instantiate(cx, impl_args),
        )?;

        // Equate IAT with the RHS of the project goal
        let inherent_args = inherent.rebase_inherent_args_onto_impl(impl_args, cx);

        // Check both where clauses on the impl and IAT
        //
        // FIXME(-Znext-solver=coinductive): I think this should be split
        // and we tag the impl bounds with `GoalSource::ImplWhereBound`?
        // Right now this includes both the impl and the assoc item where bounds,
        // and I don't think the assoc item where-bounds are allowed to be coinductive.
        //
        // Projecting to the IAT also "steps out the impl contructor", so we would have
        // to be very careful when changing the impl where-clauses to be productive.
        self.add_goals(
            GoalSource::Misc,
            cx.predicates_of(inherent.def_id)
                .iter_instantiated(cx, inherent_args)
                .map(|pred| goal.with(cx, pred)),
        );

        let normalized = if inherent.kind(cx).is_type() {
            cx.type_of(inherent.def_id).instantiate(cx, inherent_args).into()
        } else {
            // FIXME(mgca): Properly handle IACs in the type system
            panic!("normalizing inherent associated consts in the type system is unsupported");
        };
        self.instantiate_normalizes_to_term(goal, normalized);
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}
