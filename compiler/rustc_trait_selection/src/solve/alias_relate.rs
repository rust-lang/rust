//! Implements the `AliasRelate` goal, which is used when unifying aliases.
//! Doing this via a separate goal is called "deferred alias relation" and part
//! of our more general approach to "lazy normalization".
//!
//! This is done by first normalizing both sides of the goal, ending up in
//! either a concrete type, rigid alias, or an infer variable.
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
//!
//! Subtle: when relating an opaque to another type, we emit a
//! `NormalizesTo(opaque, ?fresh_var)` goal when trying to normalize the opaque.
//! This nested goal starts out as ambiguous and does not actually define the opaque.
//! However, if `?fresh_var` ends up geteting equated to another type, we retry the
//! `NormalizesTo` goal, at which point the opaque is actually defined.

use super::{EvalCtxt, GoalSource};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty::{self, Ty};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_alias_relate_goal(
        &mut self,
        goal: Goal<'tcx, (ty::Term<'tcx>, ty::Term<'tcx>, ty::AliasRelationDirection)>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let Goal { param_env, predicate: (lhs, rhs, direction) } = goal;

        let Some(lhs) = self.try_normalize_term(param_env, lhs)? else {
            return self.evaluate_added_goals_and_make_canonical_response(Certainty::OVERFLOW);
        };

        let Some(rhs) = self.try_normalize_term(param_env, rhs)? else {
            return self.evaluate_added_goals_and_make_canonical_response(Certainty::OVERFLOW);
        };

        let variance = match direction {
            ty::AliasRelationDirection::Equate => ty::Variance::Invariant,
            ty::AliasRelationDirection::Subtype => ty::Variance::Covariant,
        };

        match (lhs.to_alias_ty(tcx), rhs.to_alias_ty(tcx)) {
            (None, None) => {
                self.relate(param_env, lhs, variance, rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }

            (Some(alias), None) => {
                self.relate_rigid_alias_non_alias(param_env, alias, variance, rhs)
            }
            (None, Some(alias)) => self.relate_rigid_alias_non_alias(
                param_env,
                alias,
                variance.xform(ty::Variance::Contravariant),
                lhs,
            ),

            (Some(alias_lhs), Some(alias_rhs)) => {
                self.relate(param_env, alias_lhs, variance, alias_rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }

    /// Relate a rigid alias with another type. This is the same as
    /// an ordinary relate except that we treat the outer most alias
    /// constructor as rigid.
    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn relate_rigid_alias_non_alias(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        alias: ty::AliasTy<'tcx>,
        variance: ty::Variance,
        term: ty::Term<'tcx>,
    ) -> QueryResult<'tcx> {
        // NOTE: this check is purely an optimization, the structural eq would
        // always fail if the term is not an inference variable.
        if term.is_infer() {
            let tcx = self.tcx();
            // We need to relate `alias` to `term` treating only the outermost
            // constructor as rigid, relating any contained generic arguments as
            // normal. We do this by first structurally equating the `term`
            // with the alias constructor instantiated with unconstrained infer vars,
            // and then relate this with the whole `alias`.
            //
            // Alternatively we could modify `Equate` for this case by adding another
            // variant to `StructurallyRelateAliases`.
            let identity_args = self.fresh_args_for_item(alias.def_id);
            let rigid_ctor = ty::AliasTy::new(tcx, alias.def_id, identity_args);
            self.eq_structurally_relating_aliases(param_env, term, rigid_ctor.to_ty(tcx).into())?;
            self.eq(param_env, alias, rigid_ctor)?;
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    // FIXME: This needs a name that reflects that it's okay to bottom-out with an inference var.
    /// Normalize the `term` to equate it later.
    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn try_normalize_term(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        term: ty::Term<'tcx>,
    ) -> Result<Option<ty::Term<'tcx>>, NoSolution> {
        match term.unpack() {
            ty::TermKind::Ty(ty) => {
                Ok(self.try_normalize_ty_recur(param_env, 0, ty).map(Into::into))
            }
            ty::TermKind::Const(_) => {
                if let Some(alias) = term.to_alias_ty(self.tcx()) {
                    let term = self.next_term_infer_of_kind(term);
                    self.add_goal(
                        GoalSource::Misc,
                        Goal::new(self.tcx(), param_env, ty::NormalizesTo { alias, term }),
                    );
                    self.try_evaluate_added_goals()?;
                    Ok(Some(self.resolve_vars_if_possible(term)))
                } else {
                    Ok(Some(term))
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn try_normalize_ty_recur(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        depth: usize,
        ty: Ty<'tcx>,
    ) -> Option<Ty<'tcx>> {
        if !self.tcx().recursion_limit().value_within_limit(depth) {
            return None;
        }

        let ty::Alias(_, alias) = *ty.kind() else {
            return Some(ty);
        };

        match self.commit_if_ok(|this| {
            let normalized_ty = this.next_ty_infer();
            let normalizes_to_goal = Goal::new(
                this.tcx(),
                param_env,
                ty::NormalizesTo { alias, term: normalized_ty.into() },
            );
            this.add_goal(GoalSource::Misc, normalizes_to_goal);
            this.try_evaluate_added_goals()?;
            Ok(this.resolve_vars_if_possible(normalized_ty))
        }) {
            Ok(ty) => self.try_normalize_ty_recur(param_env, depth + 1, ty),
            Err(NoSolution) => Some(ty),
        }
    }
}
