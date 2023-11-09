//! Implements the `AliasRelate` goal, which is used when unifying aliases.
//! Doing this via a separate goal is called "deferred alias relation" and part
//! of our more general approach to "lazy normalization".
//!
//! This goal, e.g. `A alias-relate B`, may be satisfied by one of three branches:
//! * normalizes-to: If `A` is a projection, we can prove the equivalent
//!   projection predicate with B as the right-hand side of the projection.
//!   This goal is computed in both directions, if both are aliases.
//! * subst-relate: Equate `A` and `B` by their substs, if they're both
//!   aliases with the same def-id.
//! * bidirectional-normalizes-to: If `A` and `B` are both projections, and both
//!   may apply, then we can compute the "intersection" of both normalizes-to by
//!   performing them together. This is used specifically to resolve ambiguities.
use super::EvalCtxt;
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
                    self.add_goal(Goal::new(
                        self.tcx(),
                        param_env,
                        ty::ProjectionPredicate { projection_ty: alias, term },
                    ));
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
        self.add_goal(Goal::new(
            self.tcx(),
            param_env,
            ty::ProjectionPredicate { projection_ty: opaque, term },
        ));
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
