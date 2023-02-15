use std::mem;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::{
    query::NoSolution, FulfillmentError, FulfillmentErrorCode, MismatchedProjectionTypes,
    PredicateObligation, SelectionError, TraitEngine,
};
use rustc_middle::ty;
use rustc_middle::ty::error::{ExpectedFound, TypeError};

use super::{Certainty, InferCtxtEvalExt};

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

    fn collect_remaining_errors(&mut self) -> Vec<FulfillmentError<'tcx>> {
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
                            code: match goal.predicate.kind().skip_binder() {
                                ty::PredicateKind::Clause(ty::Clause::Projection(_)) => {
                                    FulfillmentErrorCode::CodeProjectionError(
                                        // FIXME: This could be a `Sorts` if the term is a type
                                        MismatchedProjectionTypes { err: TypeError::Mismatch },
                                    )
                                }
                                ty::PredicateKind::AliasEq(_, _) => {
                                    FulfillmentErrorCode::CodeProjectionError(
                                        MismatchedProjectionTypes { err: TypeError::Mismatch },
                                    )
                                }
                                ty::PredicateKind::Subtype(pred) => {
                                    let (a, b) = infcx.instantiate_binder_with_placeholders(
                                        goal.predicate.kind().rebind((pred.a, pred.b)),
                                    );
                                    let expected_found = ExpectedFound::new(true, a, b);
                                    FulfillmentErrorCode::CodeSubtypeError(
                                        expected_found,
                                        TypeError::Sorts(expected_found),
                                    )
                                }
                                ty::PredicateKind::Coerce(pred) => {
                                    let (a, b) = infcx.instantiate_binder_with_placeholders(
                                        goal.predicate.kind().rebind((pred.a, pred.b)),
                                    );
                                    let expected_found = ExpectedFound::new(false, a, b);
                                    FulfillmentErrorCode::CodeSubtypeError(
                                        expected_found,
                                        TypeError::Sorts(expected_found),
                                    )
                                }
                                ty::PredicateKind::ConstEquate(a, b) => {
                                    let (a, b) = infcx.instantiate_binder_with_placeholders(
                                        goal.predicate.kind().rebind((a, b)),
                                    );
                                    let expected_found = ExpectedFound::new(true, a, b);
                                    FulfillmentErrorCode::CodeConstEquateError(
                                        expected_found,
                                        TypeError::ConstMismatch(expected_found),
                                    )
                                }
                                ty::PredicateKind::Clause(_)
                                | ty::PredicateKind::WellFormed(_)
                                | ty::PredicateKind::ObjectSafe(_)
                                | ty::PredicateKind::ClosureKind(_, _, _)
                                | ty::PredicateKind::ConstEvaluatable(_)
                                | ty::PredicateKind::TypeWellFormedFromEnv(_)
                                | ty::PredicateKind::Ambiguous => {
                                    FulfillmentErrorCode::CodeSelectionError(
                                        SelectionError::Unimplemented,
                                    )
                                }
                            },
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

    fn drain_unstalled_obligations(
        &mut self,
        _: &InferCtxt<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
        unimplemented!()
    }
}
