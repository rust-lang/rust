//! Fulfill loop for next-solver.

use std::marker::PhantomData;
use std::mem;
use std::ops::ControlFlow;
use std::vec::ExtractIf;

use rustc_next_trait_solver::delegate::SolverDelegate;
use rustc_next_trait_solver::solve::{
    GoalEvaluation, GoalStalledOn, HasChanged, SolverDelegateEvalExt,
};
use rustc_type_ir::Interner;
use rustc_type_ir::inherent::Span as _;
use rustc_type_ir::solve::{Certainty, NoSolution};

use crate::next_solver::infer::InferCtxt;
use crate::next_solver::infer::traits::{PredicateObligation, PredicateObligations};
use crate::next_solver::{DbInterner, SolverContext, Span, TypingMode};

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
pub struct FulfillmentCtxt<'db> {
    obligations: ObligationStorage<'db>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
}

#[derive(Default, Debug)]
struct ObligationStorage<'db> {
    /// Obligations which resulted in an overflow in fulfillment itself.
    ///
    /// We cannot eagerly return these as error so we instead store them here
    /// to avoid recomputing them each time `select_where_possible` is called.
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

    fn has_pending_obligations(&self) -> bool {
        !self.pending.is_empty() || !self.overflowed.is_empty()
    }

    fn clone_pending(&self) -> PredicateObligations<'db> {
        let mut obligations: PredicateObligations<'db> =
            self.pending.iter().map(|(o, _)| o.clone()).collect();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn drain_pending(
        &mut self,
        cond: impl Fn(&PredicateObligation<'db>) -> bool,
    ) -> PendingObligations<'db> {
        let (not_stalled, pending) =
            mem::take(&mut self.pending).into_iter().partition(|(o, _)| cond(o));
        self.pending = pending;
        not_stalled
    }

    fn on_fulfillment_overflow(&mut self, infcx: &InferCtxt<'db>) {
        infcx.probe(|_| {
            // IMPORTANT: we must not use solve any inference variables in the obligations
            // as this is all happening inside of a probe. We use a probe to make sure
            // we get all obligations involved in the overflow. We pretty much check: if
            // we were to do another step of `select_where_possible`, which goals would
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
    #[tracing::instrument(level = "trace", skip(self, infcx))]
    pub(crate) fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'db>,
        obligation: PredicateObligation<'db>,
    ) {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        self.obligations.register(obligation, None);
    }

    pub(crate) fn collect_remaining_errors(
        &mut self,
        infcx: &InferCtxt<'db>,
    ) -> Vec<NextSolverError<'db>> {
        self.obligations
            .pending
            .drain(..)
            .map(|(obligation, _)| NextSolverError::Ambiguity(obligation))
            .chain(self.obligations.overflowed.drain(..).map(NextSolverError::Overflow))
            .collect()
    }

    pub(crate) fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'db>,
    ) -> Vec<NextSolverError<'db>> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = Vec::new();
        loop {
            let mut any_changed = false;
            for (mut obligation, stalled_on) in self.obligations.drain_pending(|_| true) {
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
                        Certainty::Maybe(_) => {
                            self.obligations.register(obligation, None);
                        }
                    }
                    continue;
                }

                let result = delegate.evaluate_root_goal(goal, Span::dummy(), stalled_on);
                let GoalEvaluation { certainty, has_changed, stalled_on } = match result {
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
                    Certainty::Maybe(_) => self.obligations.register(obligation, stalled_on),
                }
            }

            if !any_changed {
                break;
            }
        }

        errors
    }

    pub(crate) fn select_all_or_error(
        &mut self,
        infcx: &InferCtxt<'db>,
    ) -> Vec<NextSolverError<'db>> {
        let errors = self.select_where_possible(infcx);
        if !errors.is_empty() {
            return errors;
        }

        self.collect_remaining_errors(infcx)
    }

    fn has_pending_obligations(&self) -> bool {
        self.obligations.has_pending_obligations()
    }

    fn pending_obligations(&self) -> PredicateObligations<'db> {
        self.obligations.clone_pending()
    }
}

#[derive(Debug)]
pub enum NextSolverError<'db> {
    TrueError(PredicateObligation<'db>),
    Ambiguity(PredicateObligation<'db>),
    Overflow(PredicateObligation<'db>),
}
