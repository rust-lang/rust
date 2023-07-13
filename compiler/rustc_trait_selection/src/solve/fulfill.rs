use std::mem;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::solve::MaybeCause;
use rustc_infer::traits::Obligation;
use rustc_infer::traits::{
    query::NoSolution, FulfillmentError, FulfillmentErrorCode, MismatchedProjectionTypes,
    PredicateObligation, SelectionError, TraitEngine,
};
use rustc_middle::ty;
use rustc_middle::ty::error::{ExpectedFound, TypeError};

use super::eval_ctxt::GenerateProofTree;
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

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
}

impl<'tcx> FulfillmentCtxt<'tcx> {
    pub fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentCtxt<'tcx> {
        FulfillmentCtxt { obligations: Vec::new(), usable_in_snapshot: infcx.num_open_snapshots() }
    }
}

impl<'tcx> TraitEngine<'tcx> for FulfillmentCtxt<'tcx> {
    #[instrument(level = "debug", skip(self, infcx))]
    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        self.obligations.push(obligation);
    }

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        self.obligations
            .drain(..)
            .map(|obligation| {
                let code = infcx.probe(|_| {
                    match infcx
                        .evaluate_root_goal(obligation.clone().into(), GenerateProofTree::IfEnabled)
                        .0
                    {
                        Ok((_, Certainty::Maybe(MaybeCause::Ambiguity), _)) => {
                            FulfillmentErrorCode::CodeAmbiguity { overflow: false }
                        }
                        Ok((_, Certainty::Maybe(MaybeCause::Overflow), _)) => {
                            FulfillmentErrorCode::CodeAmbiguity { overflow: true }
                        }
                        Ok((_, Certainty::Yes, _)) => {
                            bug!("did not expect successful goal when collecting ambiguity errors")
                        }
                        Err(_) => {
                            bug!("did not expect selection error when collecting ambiguity errors")
                        }
                    }
                });

                FulfillmentError {
                    obligation: obligation.clone(),
                    code,
                    root_obligation: obligation,
                }
            })
            .collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = Vec::new();
        for i in 0.. {
            if !infcx.tcx.recursion_limit().value_within_limit(i) {
                unimplemented!("overflowed on pending obligations: {:?}", self.obligations);
            }

            let mut has_changed = false;
            for obligation in mem::take(&mut self.obligations) {
                let goal = obligation.clone().into();
                let (changed, certainty, nested_goals) =
                    match infcx.evaluate_root_goal(goal, GenerateProofTree::IfEnabled).0 {
                        Ok(result) => result,
                        Err(NoSolution) => {
                            errors.push(FulfillmentError {
                                obligation: obligation.clone(),
                                code: match goal.predicate.kind().skip_binder() {
                                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(_)) => {
                                        FulfillmentErrorCode::CodeProjectionError(
                                            // FIXME: This could be a `Sorts` if the term is a type
                                            MismatchedProjectionTypes { err: TypeError::Mismatch },
                                        )
                                    }
                                    ty::PredicateKind::AliasRelate(_, _, _) => {
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
                                    ty::PredicateKind::Clause(_)
                                    | ty::PredicateKind::ObjectSafe(_)
                                    | ty::PredicateKind::ClosureKind(_, _, _)
                                    | ty::PredicateKind::Ambiguous => {
                                        FulfillmentErrorCode::CodeSelectionError(
                                            SelectionError::Unimplemented,
                                        )
                                    }
                                    ty::PredicateKind::ConstEquate(..) => {
                                        bug!("unexpected goal: {goal:?}")
                                    }
                                },
                                root_obligation: obligation,
                            });
                            continue;
                        }
                    };
                // Push any nested goals that we get from unifying our canonical response
                // with our obligation onto the fulfillment context.
                self.obligations.extend(nested_goals.into_iter().map(|goal| {
                    Obligation::new(
                        infcx.tcx,
                        obligation.cause.clone(),
                        goal.param_env,
                        goal.predicate,
                    )
                }));
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
        std::mem::take(&mut self.obligations)
    }
}
