use super::{EvalCtxt, SolverMode};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::ty;

/// We may need to invert the alias relation direction if dealing an alias on the RHS.
#[derive(Debug)]
enum Invert {
    No,
    Yes,
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_alias_relate_goal(
        &mut self,
        goal: Goal<'tcx, (ty::Term<'tcx>, ty::Term<'tcx>, ty::AliasRelationDirection)>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let Goal { param_env, predicate: (lhs, rhs, direction) } = goal;
        if lhs.is_infer() || rhs.is_infer() {
            bug!(
                "`AliasRelate` goal with an infer var on lhs or rhs which should have been instantiated"
            );
        }

        match (lhs.to_alias_ty(tcx), rhs.to_alias_ty(tcx)) {
            (None, None) => bug!("`AliasRelate` goal without an alias on either lhs or rhs"),

            // RHS is not a projection, only way this is true is if LHS normalizes-to RHS
            (Some(alias_lhs), None) => self.assemble_normalizes_to_candidate(
                param_env,
                alias_lhs,
                rhs,
                direction,
                Invert::No,
            ),

            // LHS is not a projection, only way this is true is if RHS normalizes-to LHS
            (None, Some(alias_rhs)) => self.assemble_normalizes_to_candidate(
                param_env,
                alias_rhs,
                lhs,
                direction,
                Invert::Yes,
            ),

            (Some(alias_lhs), Some(alias_rhs)) => {
                debug!("both sides are aliases");

                let mut candidates = Vec::new();
                // LHS normalizes-to RHS
                candidates.extend(self.assemble_normalizes_to_candidate(
                    param_env,
                    alias_lhs,
                    rhs,
                    direction,
                    Invert::No,
                ));
                // RHS normalizes-to RHS
                candidates.extend(self.assemble_normalizes_to_candidate(
                    param_env,
                    alias_rhs,
                    lhs,
                    direction,
                    Invert::Yes,
                ));
                // Relate via substs
                let subst_relate_response = self
                    .assemble_subst_relate_candidate(param_env, alias_lhs, alias_rhs, direction);
                candidates.extend(subst_relate_response);
                debug!(?candidates);

                if let Some(merged) = self.try_merge_responses(&candidates) {
                    Ok(merged)
                } else {
                    // When relating two aliases and we have ambiguity, we prefer
                    // relating the generic arguments of the aliases over normalizing
                    // them. This is necessary for inference during typeck.
                    //
                    // As this is incomplete, we must not do so during coherence.
                    match self.solver_mode() {
                        SolverMode::Normal => {
                            if let Ok(subst_relate_response) = subst_relate_response {
                                Ok(subst_relate_response)
                            } else if let Ok(bidirectional_normalizes_to_response) = self
                                .assemble_bidirectional_normalizes_to_candidate(
                                    param_env, lhs, rhs, direction,
                                )
                            {
                                Ok(bidirectional_normalizes_to_response)
                            } else {
                                self.flounder(&candidates)
                            }
                        }
                        SolverMode::Coherence => self.flounder(&candidates),
                    }
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn assemble_normalizes_to_candidate(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        alias: ty::AliasTy<'tcx>,
        other: ty::Term<'tcx>,
        direction: ty::AliasRelationDirection,
        invert: Invert,
    ) -> QueryResult<'tcx> {
        self.probe_candidate("normalizes-to").enter(|ecx| {
            ecx.normalizes_to_inner(param_env, alias, other, direction, invert)?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn normalizes_to_inner(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        alias: ty::AliasTy<'tcx>,
        other: ty::Term<'tcx>,
        direction: ty::AliasRelationDirection,
        invert: Invert,
    ) -> Result<(), NoSolution> {
        let other = match direction {
            // This is purely an optimization.
            ty::AliasRelationDirection::Equate => other,

            ty::AliasRelationDirection::Subtype => {
                let fresh = self.next_term_infer_of_kind(other);
                let (sub, sup) = match invert {
                    Invert::No => (fresh, other),
                    Invert::Yes => (other, fresh),
                };
                self.sub(param_env, sub, sup)?;
                fresh
            }
        };
        self.add_goal(Goal::new(
            self.tcx(),
            param_env,
            ty::Binder::dummy(ty::ProjectionPredicate { projection_ty: alias, term: other }),
        ));

        Ok(())
    }

    fn assemble_subst_relate_candidate(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        alias_lhs: ty::AliasTy<'tcx>,
        alias_rhs: ty::AliasTy<'tcx>,
        direction: ty::AliasRelationDirection,
    ) -> QueryResult<'tcx> {
        self.probe_candidate("substs relate").enter(|ecx| {
            match direction {
                ty::AliasRelationDirection::Equate => {
                    ecx.eq(param_env, alias_lhs, alias_rhs)?;
                }
                ty::AliasRelationDirection::Subtype => {
                    ecx.sub(param_env, alias_lhs, alias_rhs)?;
                }
            }

            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn assemble_bidirectional_normalizes_to_candidate(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: ty::Term<'tcx>,
        rhs: ty::Term<'tcx>,
        direction: ty::AliasRelationDirection,
    ) -> QueryResult<'tcx> {
        self.probe_candidate("bidir normalizes-to").enter(|ecx| {
            ecx.normalizes_to_inner(
                param_env,
                lhs.to_alias_ty(ecx.tcx()).unwrap(),
                rhs,
                direction,
                Invert::No,
            )?;
            ecx.normalizes_to_inner(
                param_env,
                rhs.to_alias_ty(ecx.tcx()).unwrap(),
                lhs,
                direction,
                Invert::Yes,
            )?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }
}
