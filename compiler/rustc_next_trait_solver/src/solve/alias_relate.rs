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

use rustc_type_ir::data_structures::HashSet;
use rustc_type_ir::inherent::*;
use rustc_type_ir::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use rustc_type_ir::{self as ty, Interner};
use tracing::{instrument, trace};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, NoSolution, QueryResult};

enum IgnoreAliases {
    Yes,
    No,
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_alias_relate_goal(
        &mut self,
        goal: Goal<I, (I::Term, I::Term, ty::AliasRelationDirection)>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let Goal { param_env, predicate: (lhs, rhs, direction) } = goal;
        debug_assert!(lhs.to_alias_term().is_some() || rhs.to_alias_term().is_some());

        if self.alias_cannot_name_placeholder_in_rigid(param_env, lhs, rhs)
            || self.alias_cannot_name_placeholder_in_rigid(param_env, rhs, lhs)
        {
            return Err(NoSolution);
        }

        // Structurally normalize the lhs.
        let lhs = if let Some(alias) = lhs.to_alias_term() {
            let term = self.next_term_infer_of_kind(lhs);
            self.add_normalizes_to_goal(goal.with(cx, ty::NormalizesTo { alias, term }));
            term
        } else {
            lhs
        };

        // Structurally normalize the rhs.
        let rhs = if let Some(alias) = rhs.to_alias_term() {
            let term = self.next_term_infer_of_kind(rhs);
            self.add_normalizes_to_goal(goal.with(cx, ty::NormalizesTo { alias, term }));
            term
        } else {
            rhs
        };

        // Add a `make_canonical_response` probe step so that we treat this as
        // a candidate, even if `try_evaluate_added_goals` bails due to an error.
        // It's `Certainty::AMBIGUOUS` because this candidate is not "finished",
        // since equating the normalized terms will lead to additional constraints.
        self.inspect.make_canonical_response(Certainty::AMBIGUOUS);

        // Apply the constraints.
        self.try_evaluate_added_goals()?;
        let lhs = self.resolve_vars_if_possible(lhs);
        let rhs = self.resolve_vars_if_possible(rhs);
        trace!(?lhs, ?rhs);

        let variance = match direction {
            ty::AliasRelationDirection::Equate => ty::Invariant,
            ty::AliasRelationDirection::Subtype => ty::Covariant,
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
                    variance.xform(ty::Contravariant),
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

    /// In case a rigid term refers to a placeholder which is not referenced by the
    /// alias, the alias cannot be normalized to that rigid term unless it contains
    /// either inference variables or these placeholders are referenced in a term
    /// of a `Projection`-clause in the environment.
    fn alias_cannot_name_placeholder_in_rigid(
        &mut self,
        param_env: I::ParamEnv,
        rigid_term: I::Term,
        alias: I::Term,
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
        for clause in param_env.caller_bounds().iter() {
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
                | ty::ClauseKind::HostEffect(_)
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
        term: I::Term,
        ignore_aliases: IgnoreAliases,
    ) -> HashSet<I::Term> {
        // Fast path to avoid walking the term.
        if !term.has_placeholders() {
            return Default::default();
        }

        struct PlaceholderCollector<I: Interner> {
            ignore_aliases: IgnoreAliases,
            placeholders: HashSet<I::Term>,
        }
        impl<I: Interner> TypeVisitor<I> for PlaceholderCollector<I> {
            type Result = ();

            fn visit_ty(&mut self, t: I::Ty) {
                match t.kind() {
                    ty::Placeholder(_) => drop(self.placeholders.insert(t.into())),
                    ty::Alias(..) if matches!(self.ignore_aliases, IgnoreAliases::Yes) => {}
                    _ => t.super_visit_with(self),
                }
            }

            fn visit_const(&mut self, ct: I::Const) {
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
