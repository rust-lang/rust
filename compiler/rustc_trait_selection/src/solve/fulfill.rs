use std::mem;

use super::{Certainty, InferCtxtEvalExt};
use rustc_infer::{
    infer::InferCtxt,
    traits::{
        query::NoSolution, FulfillmentError, FulfillmentErrorCode, PredicateObligation,
        SelectionError, TraitEngine,
    },
};

/// A trait engine using the new trait solver.
///
/// This is mostly identical to how `evaluate_all` works inside of the
/// solver, except that the requirements are slightly different.
///
/// Unlike `evaluate_all` it is possible to add new obligations later on
/// and we also have to track diagnostics information by using `Obligation`
/// instead of `Goal`.
///
/// It is also likely that we want to use slightly different datastructures
/// here as this will have to deal with far more root goals than `evaluate_all`.
pub struct FulfillmentCtxt<'tcx> {
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> FulfillmentCtxt<'tcx> {
    pub fn new() -> FulfillmentCtxt<'tcx> {
        FulfillmentCtxt { obligations: Vec::new() }
    }
}

impl<'tcx> TraitEngine<'tcx> for FulfillmentCtxt<'tcx> {
    fn register_predicate_obligation(
        &mut self,
        _infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        self.obligations.push(obligation);
    }

    fn select_all_or_error(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let errors = self.select_where_possible(infcx);
        if !errors.is_empty() {
            return errors;
        }

        self.obligations
            .drain(..)
            .map(|obligation| FulfillmentError {
                obligation: obligation.clone(),
                code: FulfillmentErrorCode::CodeAmbiguity,
                root_obligation: obligation,
            })
            .collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let mut errors = Vec::new();
        for i in 0.. {
            if !infcx.tcx.recursion_limit().value_within_limit(i) {
                unimplemented!("overflowed on pending obligations: {:?}", self.obligations);
            }

            let mut has_changed = false;
            for obligation in mem::take(&mut self.obligations) {
                let goal = obligation.clone().into();
                let (changed, certainty) = match infcx.evaluate_root_goal(goal) {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(FulfillmentError {
                            obligation: obligation.clone(),
                            code: FulfillmentErrorCode::CodeSelectionError(
                                SelectionError::Unimplemented,
                            ),
                            root_obligation: obligation,
                        });
                        continue;
                    }
                };

                has_changed |= changed;
                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => self.obligations.push(obligation),
                }
            }

            if !has_changed {
                break;
            }
        }

        errors
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.clone()
    }
}
