use crate::traits::{
    Environment,
    InEnvironment,
    TraitEngine,
    ObligationCause,
    PredicateObligation,
    FulfillmentError,
    FulfillmentErrorCode,
    SelectionError,
};
use crate::traits::query::NoSolution;
use crate::infer::InferCtxt;
use crate::infer::canonical::{Canonical, OriginalQueryValues};
use crate::ty::{self, Ty};
use rustc_data_structures::fx::FxHashSet;

pub type CanonicalGoal<'tcx> = Canonical<'tcx, InEnvironment<'tcx, ty::Predicate<'tcx>>>;

pub struct FulfillmentContext<'tcx> {
    obligations: FxHashSet<InEnvironment<'tcx, PredicateObligation<'tcx>>>,
}

impl FulfillmentContext<'tcx> {
    crate fn new() -> Self {
        FulfillmentContext {
            obligations: FxHashSet::default(),
        }
    }
}

fn in_environment(
    infcx: &InferCtxt<'_, 'tcx>,
    obligation: PredicateObligation<'tcx>,
) -> InEnvironment<'tcx, PredicateObligation<'tcx>> {
    assert!(!infcx.is_in_snapshot());
    let obligation = infcx.resolve_vars_if_possible(&obligation);

    let environment = match obligation.param_env.def_id {
        Some(def_id) => infcx.tcx.environment(def_id),
        None if obligation.param_env.caller_bounds.is_empty() => Environment {
            clauses: ty::List::empty(),
        },
        _ => bug!("non-empty `ParamEnv` with no def-id"),
    };

    InEnvironment {
        environment,
        goal: obligation,
    }
}

impl TraitEngine<'tcx> for FulfillmentContext<'tcx> {
    fn normalize_projection_type(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        _param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        _cause: ObligationCause<'tcx>,
    ) -> Ty<'tcx> {
        infcx.tcx.mk_ty(ty::Projection(projection_ty))
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        self.obligations.insert(in_environment(infcx, obligation));
    }

    fn select_all_or_error(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        self.select_where_possible(infcx)?;

        if self.obligations.is_empty() {
            Ok(())
        } else {
            let errors = self.obligations.iter()
                .map(|obligation| FulfillmentError {
                    obligation: obligation.goal.clone(),
                    code: FulfillmentErrorCode::CodeAmbiguity,
                })
                .collect();
            Err(errors)
        }
    }

    fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        let mut errors = Vec::new();
        let mut next_round = FxHashSet::default();
        let mut making_progress;

        loop {
            making_progress = false;

            // We iterate over all obligations, and record if we are able
            // to unambiguously prove at least one obligation.
            for obligation in self.obligations.drain() {
                let mut orig_values = OriginalQueryValues::default();
                let canonical_goal = infcx.canonicalize_query(&InEnvironment {
                    environment: obligation.environment,
                    goal: obligation.goal.predicate,
                }, &mut orig_values);

                match infcx.tcx.global_tcx().evaluate_goal(canonical_goal) {
                    Ok(response) => {
                        if response.is_proven() {
                            making_progress = true;

                            match infcx.instantiate_query_response_and_region_obligations(
                                &obligation.goal.cause,
                                obligation.goal.param_env,
                                &orig_values,
                                &response
                            ) {
                                Ok(infer_ok) => next_round.extend(
                                    infer_ok.obligations
                                        .into_iter()
                                        .map(|obligation| in_environment(infcx, obligation))
                                ),

                                Err(_err) => errors.push(FulfillmentError {
                                    obligation: obligation.goal,
                                    code: FulfillmentErrorCode::CodeSelectionError(
                                        SelectionError::Unimplemented
                                    ),
                                }),
                            }
                        } else {
                            // Ambiguous: retry at next round.
                            next_round.insert(obligation);
                        }
                    }

                    Err(NoSolution) => errors.push(FulfillmentError {
                        obligation: obligation.goal,
                        code: FulfillmentErrorCode::CodeSelectionError(
                            SelectionError::Unimplemented
                        ),
                    })
                }
            }
            next_round = std::mem::replace(&mut self.obligations, next_round);

            if !making_progress {
                break;
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.iter().map(|obligation| obligation.goal.clone()).collect()
    }
}
