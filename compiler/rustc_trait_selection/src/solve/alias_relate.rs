//! Implements the `AliasRelate` goal, which is used when unifying aliases.
//! Doing this via a separate goal is called "deferred alias relation" and part
//! of our more general approach to "lazy normalization".
//!
//! This is done by first structurally normalizing both sides of the goal, ending
//! up in either a concrete type, rigid alias, or an infer variable.
//! These are related further according to the rules below:
//!
//! (1.) If we end up with two rigid aliases, then we relate them structurally.
//!
//! (2.) If we end up with an infer var and a rigid alias, then we instantiate
//! the infer var with the constructor of the alias and then recursively relate
//! the terms.
//!
//! (3.) Otherwise, if we end with two rigid (non-projection) or infer types,
//! relate them structurally.

use super::EvalCtxt;
use rustc_data_structures::fx::FxHashSet;
use rustc_infer::infer::InferCtxt;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};

impl<'tcx> EvalCtxt<'_, InferCtxt<'tcx>> {
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_alias_relate_goal(
        &mut self,
        goal: Goal<'tcx, (ty::Term<'tcx>, ty::Term<'tcx>, ty::AliasRelationDirection)>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let Goal { param_env, predicate: (lhs, rhs, direction) } = goal;
        debug_assert!(lhs.to_alias_term().is_some() || rhs.to_alias_term().is_some());

        if self.fast_reject_unnameable_rigid_term(param_env, lhs, rhs)
            || self.fast_reject_unnameable_rigid_term(param_env, rhs, lhs)
        {
            return Err(NoSolution);
        }

        if self.fast_reject_unnameable_rigid_term(param_env, lhs, rhs)
            || self.fast_reject_unnameable_rigid_term(param_env, rhs, lhs)
        {
            return Err(NoSolution);
        }

        // Structurally normalize the lhs.
        let lhs = if let Some(alias) = lhs.to_alias_term() {
            let term = self.next_term_infer_of_kind(lhs);
            self.add_normalizes_to_goal(goal.with(tcx, ty::NormalizesTo { alias, term }));
            term
        } else {
            lhs
        };

        // Structurally normalize the rhs.
        let rhs = if let Some(alias) = rhs.to_alias_term() {
            let term = self.next_term_infer_of_kind(rhs);
            self.add_normalizes_to_goal(goal.with(tcx, ty::NormalizesTo { alias, term }));
            term
        } else {
            rhs
        };

        // Apply the constraints.
        self.try_evaluate_added_goals()?;
        let lhs = self.resolve_vars_if_possible(lhs);
        let rhs = self.resolve_vars_if_possible(rhs);
        trace!(?lhs, ?rhs);

        let variance = match direction {
            ty::AliasRelationDirection::Equate => ty::Variance::Invariant,
            ty::AliasRelationDirection::Subtype => ty::Variance::Covariant,
        };
        match (lhs.to_alias_term(), rhs.to_alias_term()) {
            (None, None) => {
                self.relate(param_env, lhs, variance, rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }

            (Some(alias), None) => {
                self.relate_rigid_alias_non_alias(param_env, alias, variance, rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            (None, Some(alias)) => {
                self.relate_rigid_alias_non_alias(
                    param_env,
                    alias,
                    variance.xform(ty::Variance::Contravariant),
                    lhs,
                )?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }

            (Some(alias_lhs), Some(alias_rhs)) => {
                self.relate(param_env, alias_lhs, variance, alias_rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }
}

enum IgnoreAliases {
    Yes,
    No,
}

impl<'tcx> EvalCtxt<'_, InferCtxt<'tcx>> {
    /// In case a rigid term refers to a placeholder which is not referenced by the
    /// alias, the alias cannot be normalized to that rigid term unless it contains
    /// either inference variables or these placeholders are referenced in a term
    /// of a `Projection`-clause in the environment.
    fn fast_reject_unnameable_rigid_term(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        rigid_term: ty::Term<'tcx>,
        alias: ty::Term<'tcx>,
    ) -> bool {
        // Check that the rigid term is actually rigid.
        if rigid_term.to_alias_term().is_some() || alias.to_alias_term().is_none() {
            return false;
        }

        // If the alias has any type or const inference variables,
        // do not try to apply the fast path as these inference variables
        // may resolve to something containing placeholders.
        if alias.has_non_region_infer() {
            return false;
        }

        let mut referenced_placeholders =
            self.collect_placeholders_in_term(rigid_term, IgnoreAliases::Yes);
        for clause in param_env.caller_bounds() {
            match clause.kind().skip_binder() {
                ty::ClauseKind::Projection(ty::ProjectionPredicate { term, .. }) => {
                    if term.has_non_region_infer() {
                        return false;
                    }

                    let env_term_placeholders =
                        self.collect_placeholders_in_term(term, IgnoreAliases::No);
                    #[allow(rustc::potential_query_instability)]
                    referenced_placeholders.retain(|p| !env_term_placeholders.contains(p));
                }
                ty::ClauseKind::Trait(_)
                | ty::ClauseKind::TypeOutlives(_)
                | ty::ClauseKind::RegionOutlives(_)
                | ty::ClauseKind::ConstArgHasType(..)
                | ty::ClauseKind::WellFormed(_)
                | ty::ClauseKind::ConstEvaluatable(_) => continue,
            }
        }

        if referenced_placeholders.is_empty() {
            return false;
        }

        let alias_placeholders = self.collect_placeholders_in_term(alias, IgnoreAliases::No);
        // If the rigid term references a placeholder not mentioned by the alias,
        // they can never unify.
        !referenced_placeholders.is_subset(&alias_placeholders)
    }

    fn collect_placeholders_in_term(
        &mut self,
        term: ty::Term<'tcx>,
        ignore_aliases: IgnoreAliases,
    ) -> FxHashSet<ty::Term<'tcx>> {
        // Fast path to avoid walking the term.
        if !term.has_placeholders() {
            return Default::default();
        }

        struct PlaceholderCollector<'tcx> {
            ignore_aliases: IgnoreAliases,
            placeholders: FxHashSet<ty::Term<'tcx>>,
        }
        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for PlaceholderCollector<'tcx> {
            type Result = ();

            fn visit_ty(&mut self, t: Ty<'tcx>) {
                match t.kind() {
                    ty::Placeholder(_) => drop(self.placeholders.insert(t.into())),
                    ty::Alias(..) if matches!(self.ignore_aliases, IgnoreAliases::Yes) => {}
                    _ => t.super_visit_with(self),
                }
            }

            fn visit_const(&mut self, ct: ty::Const<'tcx>) {
                match ct.kind() {
                    ty::ConstKind::Placeholder(_) => drop(self.placeholders.insert(ct.into())),
                    ty::ConstKind::Unevaluated(_) | ty::ConstKind::Expr(_)
                        if matches!(self.ignore_aliases, IgnoreAliases::Yes) => {}
                    _ => ct.super_visit_with(self),
                }
            }
        }

        let mut visitor = PlaceholderCollector { ignore_aliases, placeholders: Default::default() };
        term.visit_with(&mut visitor);
        visitor.placeholders
    }
}
