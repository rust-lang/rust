//! Computes a projection goal for inherent associated types,
//! `#![feature(inherent_associated_type)]`. Since HIR ty lowering already determines
//! which impl the IAT is being projected from, we just:
//! 1. instantiate generic parameters,
//! 2. equate the self type, and
//! 3. instantiate and register where clauses.

use rustc_type_ir::solve::{NoSolutionOrRerunNonErased, QueryResultOrRerunNonErased};
use rustc_type_ir::{self as ty, Interner, Unnormalized};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, GoalSource};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_inherent_associated_term(
        &mut self,
        goal: Goal<I, ty::ProjectionClause<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        let def_id = goal.predicate.projection_term.expect_inherent_def_id();
        let (inherent_kind, inherent_args) =
            self.convert_inherent_self_to_impl(goal.param_env, goal.predicate.projection_term)?;

        // Check both where clauses on the impl and IAT
        //
        // FIXME(-Znext-solver=coinductive): I think this should be split
        // and we tag the impl bounds with `GoalSource::ImplWhereBound`?
        // Right now this includes both the impl and the assoc item where bounds,
        // and I don't think the assoc item where-bounds are allowed to be coinductive.
        //
        // Projecting to the IAT also "steps out the impl constructor", so we would have
        // to be very careful when changing the impl where-clauses to be productive.
        self.add_goals(
            GoalSource::Misc,
            cx.clauses_of(def_id.into())
                .iter_instantiated(cx, inherent_args)
                .map(Unnormalized::skip_norm_wip)
                .map(|clause| goal.with(cx, clause)),
        )?;

        let normalized: I::Term = match inherent_kind {
            ty::AliasTermKind::InherentTy { def_id } => {
                let inherent = cx.type_of(def_id.into()).instantiate(cx, inherent_args);
                let inherent = self.normalize(GoalSource::Misc, goal.param_env, inherent)?;
                inherent.into()
            }
            ty::AliasTermKind::InherentConstImpl { def_id } if cx.is_type_const(def_id.into()) => {
                let inherent = cx.const_of_item(def_id.into()).instantiate(cx, inherent_args);
                let normalized_ct = self.normalize(GoalSource::Misc, goal.param_env, inherent)?;
                let normalized = normalized_ct.into();
                let term = ty::AliasTerm::new_from_args(cx, inherent_kind, inherent_args);
                self.push_const_arg_has_type_goal(goal.param_env, term, normalized)?;
                normalized
            }
            ty::AliasTermKind::InherentConstImpl { .. } => {
                let term = ty::AliasTerm::new_from_args(cx, inherent_kind, inherent_args);
                // NOTE: we intentionally pass in the `InherentConstImpl` form as the term to
                // instantiate to upon too-generic CTFE failure, as we ought to consistently compare
                // identities via `InherentConstImpl` rather than `InherentConstSelf`.
                return self.evaluate_const_and_instantiate_projection_term(
                    goal.param_env,
                    term,
                    goal.predicate.term,
                    term.expect_ct(),
                );
            }
            kind => panic!("expected inherent alias, found {kind:?}"),
        };

        self.eq(goal.param_env, goal.predicate.term, normalized)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn convert_inherent_self_to_impl(
        &mut self,
        param_env: I::ParamEnv,
        term: ty::AliasTerm<I>,
    ) -> Result<(ty::AliasTermKind<I>, I::GenericArgs), NoSolutionOrRerunNonErased> {
        match term.kind {
            ty::AliasTermKind::InherentTy { .. } | ty::AliasTermKind::InherentConstSelf { .. } => {
                let cx = self.cx();
                let def_id = term.expect_inherent_def_id();
                let impl_def_id = cx.inherent_alias_term_parent(def_id);
                let impl_args = self.fresh_args_for_item(impl_def_id.into());

                // Equate impl header and add impl where clauses
                self.eq(
                    param_env,
                    term.self_ty(),
                    cx.type_of(impl_def_id.into()).instantiate(cx, impl_args).skip_norm_wip(),
                )?;

                // Equate IAT with the RHS of the project goal
                let inherent_args = term.rebase_inherent_args_onto_impl(impl_args, cx);

                let kind = match term.kind {
                    ty::AliasTermKind::InherentTy { def_id } => {
                        ty::AliasTermKind::InherentTy { def_id }
                    }
                    ty::AliasTermKind::InherentConstSelf { def_id } => {
                        ty::AliasTermKind::InherentConstImpl { def_id }
                    }
                    _ => unreachable!(),
                };

                Ok((kind, inherent_args))
            }
            ty::AliasTermKind::InherentConstImpl { .. } => Ok((term.kind, term.args)),
            kind => panic!("expected inherent alias, found {kind:?}"),
        }
    }
}
