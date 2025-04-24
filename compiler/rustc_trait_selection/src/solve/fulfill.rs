use std::marker::PhantomData;
use std::mem;
use std::ops::ControlFlow;

use rustc_data_structures::thinvec::ExtractIf;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::{
    FromSolverError, PredicateObligation, PredicateObligations, TraitEngine,
};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor, TypingMode,
};
use rustc_next_trait_solver::solve::{GenerateProofTree, HasChanged, SolverDelegateEvalExt as _};
use rustc_span::Span;
use rustc_type_ir::data_structures::DelayedSet;
use tracing::instrument;

use self::derive_errors::*;
use super::Certainty;
use super::delegate::SolverDelegate;
use super::inspect::{self, ProofTreeInferCtxtExt};
use crate::traits::{FulfillmentError, ScrubbedTraitError};

mod derive_errors;

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
pub struct FulfillmentCtxt<'tcx, E: 'tcx> {
    obligations: ObligationStorage<'tcx>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
    _errors: PhantomData<E>,
}

#[derive(Default, Debug)]
struct ObligationStorage<'tcx> {
    /// Obligations which resulted in an overflow in fulfillment itself.
    ///
    /// We cannot eagerly return these as error so we instead store them here
    /// to avoid recomputing them each time `select_where_possible` is called.
    /// This also allows us to return the correct `FulfillmentError` for them.
    overflowed: PredicateObligations<'tcx>,
    pending: PredicateObligations<'tcx>,
}

impl<'tcx> ObligationStorage<'tcx> {
    fn register(&mut self, obligation: PredicateObligation<'tcx>) {
        self.pending.push(obligation);
    }

    fn has_pending_obligations(&self) -> bool {
        !self.pending.is_empty() || !self.overflowed.is_empty()
    }

    fn clone_pending(&self) -> PredicateObligations<'tcx> {
        let mut obligations = self.pending.clone();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn drain_pending(
        &mut self,
        cond: impl Fn(&PredicateObligation<'tcx>) -> bool,
    ) -> PredicateObligations<'tcx> {
        let (unstalled, pending) = mem::take(&mut self.pending).into_iter().partition(cond);
        self.pending = pending;
        unstalled
    }

    fn on_fulfillment_overflow(&mut self, infcx: &InferCtxt<'tcx>) {
        infcx.probe(|_| {
            // IMPORTANT: we must not use solve any inference variables in the obligations
            // as this is all happening inside of a probe. We use a probe to make sure
            // we get all obligations involved in the overflow. We pretty much check: if
            // we were to do another step of `select_where_possible`, which goals would
            // change.
            // FIXME: <https://github.com/Gankra/thin-vec/pull/66> is merged, this can be removed.
            self.overflowed.extend(ExtractIf::new(&mut self.pending, |o| {
                let goal = o.as_goal();
                let result = <&SolverDelegate<'tcx>>::from(infcx)
                    .evaluate_root_goal(goal, GenerateProofTree::No, o.cause.span)
                    .0;
                matches!(result, Ok((HasChanged::Yes, _)))
            }));
        })
    }
}

impl<'tcx, E: 'tcx> FulfillmentCtxt<'tcx, E> {
    pub fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentCtxt<'tcx, E> {
        assert!(
            infcx.next_trait_solver(),
            "new trait solver fulfillment context created when \
            infcx is set up for old trait solver"
        );
        FulfillmentCtxt {
            obligations: Default::default(),
            usable_in_snapshot: infcx.num_open_snapshots(),
            _errors: PhantomData,
        }
    }

    fn inspect_evaluated_obligation(
        &self,
        infcx: &InferCtxt<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        result: &Result<(HasChanged, Certainty), NoSolution>,
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

impl<'tcx, E> TraitEngine<'tcx, E> for FulfillmentCtxt<'tcx, E>
where
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    #[instrument(level = "trace", skip(self, infcx))]
    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        self.obligations.register(obligation);
    }

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        self.obligations
            .pending
            .drain(..)
            .map(|obligation| NextSolverError::Ambiguity(obligation))
            .chain(
                self.obligations
                    .overflowed
                    .drain(..)
                    .map(|obligation| NextSolverError::Overflow(obligation)),
            )
            .map(|e| E::from_solver_error(infcx, e))
            .collect()
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = Vec::new();
        for i in 0.. {
            if !infcx.tcx.recursion_limit().value_within_limit(i) {
                self.obligations.on_fulfillment_overflow(infcx);
                // Only return true errors that we have accumulated while processing.
                return errors;
            }

            let mut has_changed = false;
            for obligation in self.obligations.drain_pending(|_| true) {
                let goal = obligation.as_goal();
                let result = <&SolverDelegate<'tcx>>::from(infcx)
                    .evaluate_root_goal(goal, GenerateProofTree::No, obligation.cause.span)
                    .0;
                self.inspect_evaluated_obligation(infcx, &obligation, &result);
                let (changed, certainty) = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(E::from_solver_error(
                            infcx,
                            NextSolverError::TrueError(obligation),
                        ));
                        continue;
                    }
                };

                if changed == HasChanged::Yes {
                    has_changed = true;
                }

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

    fn has_pending_obligations(&self) -> bool {
        self.obligations.has_pending_obligations()
    }

    fn pending_obligations(&self) -> PredicateObligations<'tcx> {
        self.obligations.clone_pending()
    }

    fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        let stalled_generators = match infcx.typing_mode() {
            TypingMode::Analysis { defining_opaque_types_and_generators } => {
                defining_opaque_types_and_generators
            }
            TypingMode::Coherence
            | TypingMode::Borrowck { defining_opaque_types: _ }
            | TypingMode::PostBorrowckAnalysis { defined_opaque_types: _ }
            | TypingMode::PostAnalysis => return Default::default(),
        };

        if stalled_generators.is_empty() {
            return Default::default();
        }

        self.obligations.drain_pending(|obl| {
            infcx.probe(|_| {
                infcx
                    .visit_proof_tree(
                        obl.as_goal(),
                        &mut StalledOnCoroutines {
                            stalled_generators,
                            span: obl.cause.span,
                            cache: Default::default(),
                        },
                    )
                    .is_break()
            })
        })
    }
}

/// Detect if a goal is stalled on a coroutine that is owned by the current typeck root.
///
/// This function can (erroneously) fail to detect a predicate, i.e. it doesn't need to
/// be complete. However, this will lead to ambiguity errors, so we want to make it
/// accurate.
///
/// This function can be also return false positives, which will lead to poor diagnostics
/// so we want to keep this visitor *precise* too.
struct StalledOnCoroutines<'tcx> {
    stalled_generators: &'tcx ty::List<LocalDefId>,
    span: Span,
    cache: DelayedSet<Ty<'tcx>>,
}

impl<'tcx> inspect::ProofTreeVisitor<'tcx> for StalledOnCoroutines<'tcx> {
    type Result = ControlFlow<()>;

    fn span(&self) -> rustc_span::Span {
        self.span
    }

    fn visit_goal(&mut self, inspect_goal: &super::inspect::InspectGoal<'_, 'tcx>) -> Self::Result {
        inspect_goal.goal().predicate.visit_with(self)?;

        if let Some(candidate) = inspect_goal.unique_applicable_candidate() {
            candidate.visit_nested_no_probe(self)
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for StalledOnCoroutines<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        if !self.cache.insert(ty) {
            return ControlFlow::Continue(());
        }

        if let ty::CoroutineWitness(def_id, _) = *ty.kind()
            && def_id.as_local().is_some_and(|def_id| self.stalled_generators.contains(&def_id))
        {
            return ControlFlow::Break(());
        }

        ty.super_visit_with(self)
    }
}

pub enum NextSolverError<'tcx> {
    TrueError(PredicateObligation<'tcx>),
    Ambiguity(PredicateObligation<'tcx>),
    Overflow(PredicateObligation<'tcx>),
}

impl<'tcx> FromSolverError<'tcx, NextSolverError<'tcx>> for FulfillmentError<'tcx> {
    fn from_solver_error(infcx: &InferCtxt<'tcx>, error: NextSolverError<'tcx>) -> Self {
        match error {
            NextSolverError::TrueError(obligation) => {
                fulfillment_error_for_no_solution(infcx, obligation)
            }
            NextSolverError::Ambiguity(obligation) => {
                fulfillment_error_for_stalled(infcx, obligation)
            }
            NextSolverError::Overflow(obligation) => {
                fulfillment_error_for_overflow(infcx, obligation)
            }
        }
    }
}

impl<'tcx> FromSolverError<'tcx, NextSolverError<'tcx>> for ScrubbedTraitError<'tcx> {
    fn from_solver_error(_infcx: &InferCtxt<'tcx>, error: NextSolverError<'tcx>) -> Self {
        match error {
            NextSolverError::TrueError(_) => ScrubbedTraitError::TrueError,
            NextSolverError::Ambiguity(_) | NextSolverError::Overflow(_) => {
                ScrubbedTraitError::Ambiguity
            }
        }
    }
}
