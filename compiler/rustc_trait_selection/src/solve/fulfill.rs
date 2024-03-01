use std::mem;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::solve::MaybeCause;
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
    obligations: ObligationStorage<'tcx>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
}

#[derive(Default)]
struct ObligationStorage<'tcx> {
    /// Obligations which resulted in an overflow in fulfillment itself.
    ///
    /// We cannot eagerly return these as error so we instead store them here
    /// to avoid recomputing them each time `select_where_possible` is called.
    /// This also allows us to return the correct `FulfillmentError` for them.
    overflowed: Vec<PredicateObligation<'tcx>>,
    pending: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> ObligationStorage<'tcx> {
    fn register(&mut self, obligation: PredicateObligation<'tcx>) {
        self.pending.push(obligation);
    }

    fn clone_pending(&self) -> Vec<PredicateObligation<'tcx>> {
        let mut obligations = self.pending.clone();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn take_pending(&mut self) -> Vec<PredicateObligation<'tcx>> {
        let mut obligations = mem::take(&mut self.pending);
        obligations.extend(self.overflowed.drain(..));
        obligations
    }

    fn unstalled_for_select(&mut self) -> impl Iterator<Item = PredicateObligation<'tcx>> {
        mem::take(&mut self.pending).into_iter()
    }

    fn on_fulfillment_overflow(&mut self, infcx: &InferCtxt<'tcx>) {
        infcx.probe(|_| {
            // IMPORTANT: we must not use solve any inference variables in the obligations
            // as this is all happening inside of a probe. We use a probe to make sure
            // we get all obligations involved in the overflow. We pretty much check: if
            // we were to do another step of `select_where_possible`, which goals would
            // change.
            self.overflowed.extend(self.pending.extract_if(|o| {
                let goal = o.clone().into();
                let result = infcx.evaluate_root_goal(goal, GenerateProofTree::Never).0;
                match result {
                    Ok((has_changed, _)) => has_changed,
                    _ => false,
                }
            }));
        })
    }
}

impl<'tcx> FulfillmentCtxt<'tcx> {
    pub fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentCtxt<'tcx> {
        assert!(
            infcx.next_trait_solver(),
            "new trait solver fulfillment context created when \
            infcx is set up for old trait solver"
        );
        FulfillmentCtxt {
            obligations: Default::default(),
            usable_in_snapshot: infcx.num_open_snapshots(),
        }
    }

    fn inspect_evaluated_obligation(
        &self,
        infcx: &InferCtxt<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        result: &Result<(bool, Certainty), NoSolution>,
    ) {
        if let Some(inspector) = infcx.obligation_inspector.get() {
            let result = match result {
                Ok((_, c)) => Ok(*c),
                Err(NoSolution) => Err(NoSolution),
            };
            (inspector)(infcx, &obligation, result);
        }
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
        self.obligations.register(obligation);
    }

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let mut errors: Vec<_> = self
            .obligations
            .pending
            .drain(..)
            .map(|obligation| fulfillment_error_for_stalled(infcx, obligation))
            .collect();

        errors.extend(self.obligations.overflowed.drain(..).map(|obligation| FulfillmentError {
            root_obligation: obligation.clone(),
            code: FulfillmentErrorCode::Ambiguity { overflow: Some(true) },
            obligation,
        }));

        errors
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = Vec::new();
        for i in 0.. {
            if !infcx.tcx.recursion_limit().value_within_limit(i) {
                self.obligations.on_fulfillment_overflow(infcx);
                // Only return true errors that we have accumulated while processing.
                return errors;
            }

            let mut has_changed = false;
            for obligation in self.obligations.unstalled_for_select() {
                let goal = obligation.clone().into();
                let result = infcx.evaluate_root_goal(goal, GenerateProofTree::IfEnabled).0;
                self.inspect_evaluated_obligation(infcx, &obligation, &result);
                let (changed, certainty) = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(fulfillment_error_for_no_solution(infcx, obligation));
                        continue;
                    }
                };
                has_changed |= changed;
                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => self.obligations.register(obligation),
                }
            }

            if !has_changed {
                break;
            }
        }

        errors
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.clone_pending()
    }

    fn drain_unstalled_obligations(
        &mut self,
        _: &InferCtxt<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.take_pending()
    }
}

fn fulfillment_error_for_no_solution<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    let code = match obligation.predicate.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::ClauseKind::Projection(_)) => {
            FulfillmentErrorCode::ProjectionError(
                // FIXME: This could be a `Sorts` if the term is a type
                MismatchedProjectionTypes { err: TypeError::Mismatch },
            )
        }
        ty::PredicateKind::NormalizesTo(..) => {
            FulfillmentErrorCode::ProjectionError(MismatchedProjectionTypes {
                err: TypeError::Mismatch,
            })
        }
        ty::PredicateKind::AliasRelate(_, _, _) => {
            FulfillmentErrorCode::ProjectionError(MismatchedProjectionTypes {
                err: TypeError::Mismatch,
            })
        }
        ty::PredicateKind::Subtype(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(true, a, b);
            FulfillmentErrorCode::SubtypeError(expected_found, TypeError::Sorts(expected_found))
        }
        ty::PredicateKind::Coerce(pred) => {
            let (a, b) = infcx.enter_forall_and_leak_universe(
                obligation.predicate.kind().rebind((pred.a, pred.b)),
            );
            let expected_found = ExpectedFound::new(false, a, b);
            FulfillmentErrorCode::SubtypeError(expected_found, TypeError::Sorts(expected_found))
        }
        ty::PredicateKind::Clause(_)
        | ty::PredicateKind::ObjectSafe(_)
        | ty::PredicateKind::Ambiguous => {
            FulfillmentErrorCode::SelectionError(SelectionError::Unimplemented)
        }
        ty::PredicateKind::ConstEquate(..) => {
            bug!("unexpected goal: {obligation:?}")
        }
    };
    FulfillmentError { root_obligation: obligation.clone(), code, obligation }
}

fn fulfillment_error_for_stalled<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: PredicateObligation<'tcx>,
) -> FulfillmentError<'tcx> {
    let code = infcx.probe(|_| {
        match infcx.evaluate_root_goal(obligation.clone().into(), GenerateProofTree::Never).0 {
            Ok((_, Certainty::Maybe(MaybeCause::Ambiguity))) => {
                FulfillmentErrorCode::Ambiguity { overflow: None }
            }
            Ok((_, Certainty::Maybe(MaybeCause::Overflow { suggest_increasing_limit }))) => {
                FulfillmentErrorCode::Ambiguity { overflow: Some(suggest_increasing_limit) }
            }
            Ok((_, Certainty::Yes)) => {
                bug!("did not expect successful goal when collecting ambiguity errors")
            }
            Err(_) => {
                bug!("did not expect selection error when collecting ambiguity errors")
            }
        }
    });

    FulfillmentError { obligation: obligation.clone(), code, root_obligation: obligation }
}
