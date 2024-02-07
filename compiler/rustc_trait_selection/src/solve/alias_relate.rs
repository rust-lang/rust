//! Implements the `AliasRelate` goal, which is used when unifying aliases.
//! Doing this via a separate goal is called "deferred alias relation" and part
//! of our more general approach to "lazy normalization".
//!
//! This is done by first normalizing both sides of the goal, ending up in
//! either a concrete type, rigid projection, opaque, or an infer variable.
//! These are related further according to the rules below:
//!
//! (1.) If we end up with a rigid projection and a rigid projection, then we
//! relate those projections structurally.
//!
//! (2.) If we end up with a rigid projection and an alias, then the opaque will
//! have its hidden type defined to be that rigid projection.
//!
//! (3.) If we end up with an opaque and an opaque, then we assemble two
//! candidates, one defining the LHS to be the hidden type of the RHS, and vice
//! versa.
//!
//! (4.) If we end up with an infer var and an opaque or rigid projection, then
//! we assign the alias to the infer var.
//!
//! (5.) If we end up with an opaque and a rigid (non-projection) type, then we
//! define the hidden type of the opaque to be the rigid type.
//!
//! (6.) Otherwise, if we end with two rigid (non-projection) or infer types,
//! relate them structurally.

use super::{EvalCtxt, GoalSource};
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty;

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
                if rhs.is_infer() {
                    self.relate(param_env, lhs, variance, rhs)?;
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                } else if alias.is_opaque(tcx) {
                    // FIXME: This doesn't account for variance.
                    self.define_opaque(param_env, alias, rhs)
                } else {
                    Err(NoSolution)
                }
            }
            (None, Some(alias)) => {
                if lhs.is_infer() {
                    self.relate(param_env, lhs, variance, rhs)?;
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                } else if alias.is_opaque(tcx) {
                    // FIXME: This doesn't account for variance.
                    self.define_opaque(param_env, alias, lhs)
                } else {
                    Err(NoSolution)
                }
            }

            (Some(alias_lhs), Some(alias_rhs)) => {
                self.relate_rigid_alias_or_opaque(param_env, alias_lhs, variance, alias_rhs)
            }
        }
    }

    // FIXME: This needs a name that reflects that it's okay to bottom-out with an inference var.
    /// Normalize the `term` to equate it later. This does not define opaque types.
    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn try_normalize_term(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        term: ty::Term<'tcx>,
    ) -> Result<Option<ty::Term<'tcx>>, NoSolution> {
        match term.unpack() {
            ty::TermKind::Ty(ty) => {
                // We do no define opaque types here but instead do so in `relate_rigid_alias_or_opaque`.
                Ok(self
                    .try_normalize_ty_recur(param_env, DefineOpaqueTypes::No, 0, ty)
                    .map(Into::into))
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

    fn define_opaque(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        opaque: ty::AliasTy<'tcx>,
        term: ty::Term<'tcx>,
    ) -> QueryResult<'tcx> {
        self.add_goal(
            GoalSource::Misc,
            Goal::new(self.tcx(), param_env, ty::NormalizesTo { alias: opaque, term }),
        );
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn relate_rigid_alias_or_opaque(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: ty::AliasTy<'tcx>,
        variance: ty::Variance,
        rhs: ty::AliasTy<'tcx>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let mut candidates = vec![];
        if lhs.is_opaque(tcx) {
            candidates.extend(
                self.probe_misc_candidate("define-lhs-opaque")
                    .enter(|ecx| ecx.define_opaque(param_env, lhs, rhs.to_ty(tcx).into())),
            );
        }

        if rhs.is_opaque(tcx) {
            candidates.extend(
                self.probe_misc_candidate("define-rhs-opaque")
                    .enter(|ecx| ecx.define_opaque(param_env, rhs, lhs.to_ty(tcx).into())),
            );
        }

        candidates.extend(self.probe_misc_candidate("args-relate").enter(|ecx| {
            ecx.relate(param_env, lhs, variance, rhs)?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        }));

        if let Some(result) = self.try_merge_responses(&candidates) {
            Ok(result)
        } else {
            self.flounder(&candidates)
        }
    }
}
