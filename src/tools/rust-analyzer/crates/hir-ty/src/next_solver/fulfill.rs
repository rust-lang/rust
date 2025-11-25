//! Fulfill loop for next-solver.

mod errors;

use std::ops::ControlFlow;

use rustc_hash::FxHashSet;
use rustc_next_trait_solver::{
    delegate::SolverDelegate,
    solve::{GoalEvaluation, GoalStalledOn, HasChanged, SolverDelegateEvalExt},
};
use rustc_type_ir::{
    Interner, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
    inherent::{IntoKind, Span as _},
    solve::{Certainty, NoSolution},
};

use crate::next_solver::{
    DbInterner, SolverContext, SolverDefId, Span, Ty, TyKind, TypingMode,
    infer::{
        InferCtxt,
        traits::{PredicateObligation, PredicateObligations},
    },
    inspect::ProofTreeVisitor,
};

type PendingObligations<'db> =
    Vec<(PredicateObligation<'db>, Option<GoalStalledOn<DbInterner<'db>>>)>;

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
#[derive(Debug, Clone)]
pub struct FulfillmentCtxt<'db> {
    obligations: ObligationStorage<'db>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    #[expect(unused)]
    usable_in_snapshot: usize,
}

#[derive(Default, Debug, Clone)]
struct ObligationStorage<'db> {
    /// Obligations which resulted in an overflow in fulfillment itself.
    ///
    /// We cannot eagerly return these as error so we instead store them here
    /// to avoid recomputing them each time `try_evaluate_obligations` is called.
    /// This also allows us to return the correct `FulfillmentError` for them.
    overflowed: Vec<PredicateObligation<'db>>,
    pending: PendingObligations<'db>,
}

impl<'db> ObligationStorage<'db> {
    fn register(
        &mut self,
        obligation: PredicateObligation<'db>,
        stalled_on: Option<GoalStalledOn<DbInterner<'db>>>,
    ) {
        self.pending.push((obligation, stalled_on));
    }

    fn clone_pending(&self) -> PredicateObligations<'db> {
        let mut obligations: PredicateObligations<'db> =
            self.pending.iter().map(|(o, _)| o.clone()).collect();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn drain_pending<'this, 'cond>(
        &'this mut self,
        cond: impl 'cond + Fn(&PredicateObligation<'db>) -> bool,
    ) -> impl Iterator<Item = (PredicateObligation<'db>, Option<GoalStalledOn<DbInterner<'db>>>)>
    {
        self.pending.extract_if(.., move |(o, _)| cond(o))
    }

    fn on_fulfillment_overflow(&mut self, infcx: &InferCtxt<'db>) {
        infcx.probe(|_| {
            // IMPORTANT: we must not use solve any inference variables in the obligations
            // as this is all happening inside of a probe. We use a probe to make sure
            // we get all obligations involved in the overflow. We pretty much check: if
            // we were to do another step of `try_evaluate_obligations`, which goals would
            // change.
            // FIXME: <https://github.com/Gankra/thin-vec/pull/66> is merged, this can be removed.
            self.overflowed.extend(
                self.pending
                    .extract_if(.., |(o, stalled_on)| {
                        let goal = o.as_goal();
                        let result = <&SolverContext<'db>>::from(infcx).evaluate_root_goal(
                            goal,
                            Span::dummy(),
                            stalled_on.take(),
                        );
                        matches!(result, Ok(GoalEvaluation { has_changed: HasChanged::Yes, .. }))
                    })
                    .map(|(o, _)| o),
            );
        })
    }
}

impl<'db> FulfillmentCtxt<'db> {
    pub fn new(infcx: &InferCtxt<'db>) -> FulfillmentCtxt<'db> {
        FulfillmentCtxt {
            obligations: Default::default(),
            usable_in_snapshot: infcx.num_open_snapshots(),
        }
    }
}

impl<'db> FulfillmentCtxt<'db> {
    #[tracing::instrument(level = "trace", skip(self, _infcx))]
    pub(crate) fn register_predicate_obligation(
        &mut self,
        _infcx: &InferCtxt<'db>,
        obligation: PredicateObligation<'db>,
    ) {
        // FIXME: See the comment in `try_evaluate_obligations()`.
        // assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        self.obligations.register(obligation, None);
    }

    pub(crate) fn register_predicate_obligations(
        &mut self,
        _infcx: &InferCtxt<'db>,
        obligations: impl IntoIterator<Item = PredicateObligation<'db>>,
    ) {
        // FIXME: See the comment in `try_evaluate_obligations()`.
        // assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        obligations.into_iter().for_each(|obligation| self.obligations.register(obligation, None));
    }

    pub(crate) fn collect_remaining_errors(
        &mut self,
        _infcx: &InferCtxt<'db>,
    ) -> Vec<NextSolverError<'db>> {
        self.obligations
            .pending
            .drain(..)
            .map(|(obligation, _)| NextSolverError::Ambiguity(obligation))
            .chain(self.obligations.overflowed.drain(..).map(NextSolverError::Overflow))
            .collect()
    }

    pub(crate) fn try_evaluate_obligations(
        &mut self,
        infcx: &InferCtxt<'db>,
    ) -> Vec<NextSolverError<'db>> {
        // FIXME(next-solver): We should bring this assertion back. Currently it panics because
        // there are places which use `InferenceTable` and open a snapshot and register obligations
        // and select. They should use a different `ObligationCtxt` instead. Then we'll be also able
        // to not put the obligations queue in `InferenceTable`'s snapshots.
        // assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = Vec::new();
        let mut obligations = Vec::new();
        loop {
            let mut any_changed = false;
            obligations.extend(self.obligations.drain_pending(|_| true));
            for (mut obligation, stalled_on) in obligations.drain(..) {
                if obligation.recursion_depth >= infcx.interner.recursion_limit() {
                    self.obligations.on_fulfillment_overflow(infcx);
                    // Only return true errors that we have accumulated while processing.
                    return errors;
                }

                let goal = obligation.as_goal();
                let delegate = <&SolverContext<'db>>::from(infcx);
                if let Some(certainty) = delegate.compute_goal_fast_path(goal, Span::dummy()) {
                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe { .. } => {
                            self.obligations.register(obligation, None);
                        }
                    }
                    continue;
                }

                let result = delegate.evaluate_root_goal(goal, Span::dummy(), stalled_on);
                let GoalEvaluation { goal: _, certainty, has_changed, stalled_on } = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(NextSolverError::TrueError(obligation));
                        continue;
                    }
                };

                if has_changed == HasChanged::Yes {
                    // We increment the recursion depth here to track the number of times
                    // this goal has resulted in inference progress. This doesn't precisely
                    // model the way that we track recursion depth in the old solver due
                    // to the fact that we only process root obligations, but it is a good
                    // approximation and should only result in fulfillment overflow in
                    // pathological cases.
                    obligation.recursion_depth += 1;
                    any_changed = true;
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe { .. } => self.obligations.register(obligation, stalled_on),
                }
            }

            if !any_changed {
                break;
            }
        }

        errors
    }

    pub(crate) fn evaluate_obligations_error_on_ambiguity(
        &mut self,
        infcx: &InferCtxt<'db>,
    ) -> Vec<NextSolverError<'db>> {
        let errors = self.try_evaluate_obligations(infcx);
        if !errors.is_empty() {
            return errors;
        }

        self.collect_remaining_errors(infcx)
    }

    pub(crate) fn pending_obligations(&self) -> PredicateObligations<'db> {
        self.obligations.clone_pending()
    }

    pub(crate) fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'db>,
    ) -> PredicateObligations<'db> {
        let stalled_coroutines = match infcx.typing_mode() {
            TypingMode::Analysis { defining_opaque_types_and_generators } => {
                defining_opaque_types_and_generators
            }
            TypingMode::Coherence
            | TypingMode::Borrowck { defining_opaque_types: _ }
            | TypingMode::PostBorrowckAnalysis { defined_opaque_types: _ }
            | TypingMode::PostAnalysis => return Default::default(),
        };
        let stalled_coroutines = stalled_coroutines.inner();

        if stalled_coroutines.is_empty() {
            return Default::default();
        }

        self.obligations
            .drain_pending(|obl| {
                infcx.probe(|_| {
                    infcx
                        .visit_proof_tree(
                            obl.as_goal(),
                            &mut StalledOnCoroutines {
                                stalled_coroutines,
                                cache: Default::default(),
                            },
                        )
                        .is_break()
                })
            })
            .map(|(o, _)| o)
            .collect()
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
pub struct StalledOnCoroutines<'a, 'db> {
    pub stalled_coroutines: &'a [SolverDefId],
    pub cache: FxHashSet<Ty<'db>>,
}

impl<'db> ProofTreeVisitor<'db> for StalledOnCoroutines<'_, 'db> {
    type Result = ControlFlow<()>;

    fn visit_goal(&mut self, inspect_goal: &super::inspect::InspectGoal<'_, 'db>) -> Self::Result {
        inspect_goal.goal().predicate.visit_with(self)?;

        if let Some(candidate) = inspect_goal.unique_applicable_candidate() {
            candidate.visit_nested_no_probe(self)
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<'db> TypeVisitor<DbInterner<'db>> for StalledOnCoroutines<'_, 'db> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'db>) -> Self::Result {
        if !self.cache.insert(ty) {
            return ControlFlow::Continue(());
        }

        if let TyKind::Coroutine(def_id, _) = ty.kind()
            && self.stalled_coroutines.contains(&def_id.into())
        {
            ControlFlow::Break(())
        } else if ty.has_coroutines() {
            ty.super_visit_with(self)
        } else {
            ControlFlow::Continue(())
        }
    }
}

#[derive(Debug)]
pub enum NextSolverError<'db> {
    TrueError(PredicateObligation<'db>),
    Ambiguity(PredicateObligation<'db>),
    Overflow(PredicateObligation<'db>),
}

impl NextSolverError<'_> {
    #[inline]
    pub fn is_true_error(&self) -> bool {
        matches!(self, NextSolverError::TrueError(_))
    }
}
