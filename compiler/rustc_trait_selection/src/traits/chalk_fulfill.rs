//! Defines a Chalk-based `TraitEngine`

use crate::infer::canonical::OriginalQueryValues;
use crate::infer::InferCtxt;
use crate::traits::query::NoSolution;
use crate::traits::{
    ChalkEnvironmentAndGoal, FulfillmentError, FulfillmentErrorCode, PredicateObligation,
    SelectionError, TraitEngine,
};
use rustc_data_structures::fx::FxIndexSet;
use rustc_middle::ty::TypeVisitableExt;

pub struct FulfillmentContext<'tcx> {
    obligations: FxIndexSet<PredicateObligation<'tcx>>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
}

impl<'tcx> FulfillmentContext<'tcx> {
    pub(super) fn new(infcx: &InferCtxt<'tcx>) -> Self {
        FulfillmentContext {
            obligations: FxIndexSet::default(),
            usable_in_snapshot: infcx.num_open_snapshots(),
        }
    }
}

impl<'tcx> TraitEngine<'tcx> for FulfillmentContext<'tcx> {
    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let obligation = infcx.resolve_vars_if_possible(obligation);

        self.obligations.insert(obligation);
    }

    fn collect_remaining_errors(
        &mut self,
        _infcx: &InferCtxt<'tcx>,
    ) -> Vec<FulfillmentError<'tcx>> {
        // any remaining obligations are errors
        self.obligations
            .iter()
            .map(|obligation| FulfillmentError {
                obligation: obligation.clone(),
                code: FulfillmentErrorCode::CodeAmbiguity { overflow: false },
                // FIXME - does Chalk have a notation of 'root obligation'?
                // This is just for diagnostics, so it's okay if this is wrong
                root_obligation: obligation.clone(),
            })
            .collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());

        let mut errors = Vec::new();
        let mut next_round = FxIndexSet::default();
        let mut making_progress;

        loop {
            making_progress = false;

            // We iterate over all obligations, and record if we are able
            // to unambiguously prove at least one obligation.
            for obligation in self.obligations.drain(..) {
                let obligation = infcx.resolve_vars_if_possible(obligation);
                let environment = obligation.param_env.caller_bounds();
                let goal = ChalkEnvironmentAndGoal { environment, goal: obligation.predicate };
                let mut orig_values = OriginalQueryValues::default();
                if goal.references_error() {
                    continue;
                }

                let canonical_goal =
                    infcx.canonicalize_query_preserving_universes(goal, &mut orig_values);

                match infcx.tcx.evaluate_goal(canonical_goal) {
                    Ok(response) => {
                        if response.is_proven() {
                            making_progress = true;

                            match infcx.instantiate_query_response_and_region_obligations(
                                &obligation.cause,
                                obligation.param_env,
                                &orig_values,
                                &response,
                            ) {
                                Ok(infer_ok) => {
                                    next_round.extend(infer_ok.obligations.into_iter().map(
                                        |obligation| infcx.resolve_vars_if_possible(obligation),
                                    ))
                                }

                                Err(_err) => errors.push(FulfillmentError {
                                    obligation: obligation.clone(),
                                    code: FulfillmentErrorCode::CodeSelectionError(
                                        SelectionError::Unimplemented,
                                    ),
                                    // FIXME - does Chalk have a notation of 'root obligation'?
                                    // This is just for diagnostics, so it's okay if this is wrong
                                    root_obligation: obligation,
                                }),
                            }
                        } else {
                            // Ambiguous: retry at next round.
                            next_round.insert(obligation);
                        }
                    }

                    Err(NoSolution) => errors.push(FulfillmentError {
                        obligation: obligation.clone(),
                        code: FulfillmentErrorCode::CodeSelectionError(
                            SelectionError::Unimplemented,
                        ),
                        // FIXME - does Chalk have a notation of 'root obligation'?
                        // This is just for diagnostics, so it's okay if this is wrong
                        root_obligation: obligation,
                    }),
                }
            }
            next_round = std::mem::replace(&mut self.obligations, next_round);

            if !making_progress {
                break;
            }
        }

        errors
    }

    fn drain_unstalled_obligations(
        &mut self,
        _: &InferCtxt<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
        unimplemented!()
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.iter().cloned().collect()
    }
}
