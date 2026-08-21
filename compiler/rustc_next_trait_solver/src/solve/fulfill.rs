use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{Certainty, Goal, NoSolution, Obligation};
use rustc_type_ir::{InferCtxtLike as _, Interner};
use thin_vec::ThinVec;

use super::fast_path::compute_goal_fast_path;
use super::{
    GoalEvaluation, GoalStalledOn, HasChanged, SolverDelegate, SolverDelegateEvalExt as _,
};

type PredicateObligation<I> = Obligation<I, <I as Interner>::Predicate>;

#[derive(Debug, Clone)]
pub enum NextSolverError<I: Interner> {
    TrueError(PredicateObligation<I>),
    Ambiguity(PredicateObligation<I>),
    Overflow(PredicateObligation<I>),
}

// FIXME: Do we need to use a `ThinVec` here?
type PendingObligations<I> = ThinVec<(PredicateObligation<I>, Option<GoalStalledOn<I>>)>;

#[derive(Debug)]
struct ObligationStorage<I: Interner> {
    /// Obligations which resulted in overflow in fulfillment itself.
    ///
    /// We cannot eagerly return these as errors, so we instead store them here
    /// to avoid recomputing them each time `try_evaluate_obligations` is called.
    /// This also allows the frontend to construct the correct error for them.
    overflowed: Vec<PredicateObligation<I>>,

    pending: PendingObligations<I>,
}

impl<I: Interner> Default for ObligationStorage<I> {
    fn default() -> Self {
        Self { overflowed: Vec::new(), pending: ThinVec::new() }
    }
}

impl<I: Interner> ObligationStorage<I> {
    fn register(
        &mut self,
        obligation: PredicateObligation<I>,
        stalled_on: Option<GoalStalledOn<I>>,
    ) {
        self.pending.push((obligation, stalled_on));
    }

    fn has_pending_obligations(&self) -> bool {
        !self.pending.is_empty() || !self.overflowed.is_empty()
    }

    fn clone_pending(&self) -> ThinVec<PredicateObligation<I>> {
        let mut obligations =
            self.pending.iter().map(|(obligation, _)| obligation.clone()).collect::<ThinVec<_>>();

        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn clone_pending_filtered<F>(&self, mut filter: F) -> ThinVec<PredicateObligation<I>>
    where
        F: FnMut(&PredicateObligation<I>, &Option<GoalStalledOn<I>>) -> bool,
    {
        let mut obligations = self
            .pending
            .iter()
            .filter_map(|(obligation, stalled_on)| {
                filter(obligation, stalled_on).then(|| obligation.clone())
            })
            .collect::<ThinVec<_>>();

        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn drain_pending<F>(&mut self, mut filter: F) -> ThinVec<PredicateObligation<I>>
    where
        F: FnMut(&PredicateObligation<I>, &Option<GoalStalledOn<I>>) -> bool,
    {
        let (drained, pending): (PendingObligations<I>, PendingObligations<I>) =
            std::mem::take(&mut self.pending)
                .into_iter()
                .partition(|(obligation, stalled_on)| filter(obligation, stalled_on));

        self.pending = pending;

        drained.into_iter().map(|(obligation, _)| obligation).collect()
    }

    #[cold]
    #[inline(never)]
    fn collect_remaining_errors<E>(
        &mut self,
        map: impl FnMut(NextSolverError<I>) -> E,
    ) -> ThinVec<E> {
        self.pending
            .drain(..)
            .map(|(obligation, _)| NextSolverError::Ambiguity(obligation))
            .chain(self.overflowed.drain(..).map(NextSolverError::Overflow))
            .map(map)
            .collect()
    }
}

/// A fulfillment engine using the new trait solver.
///
/// This is mostly identical to how `evaluate_all` works inside of the solver,
/// except that it is possible to add new obligations later and the frontend
/// needs to retain its obligation representation for diagnostics.
///
/// It is also likely that we want to use different data structures here, as
/// fulfillment deals with far more root goals than `evaluate_all`.
#[derive(Debug)]
pub struct FulfillmentCtxt<I: Interner> {
    obligations: ObligationStorage<I>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot can observe inference state which has since
    /// been rolled back.
    usable_in_snapshot: usize,
}

impl<I: Interner> FulfillmentCtxt<I> {
    pub fn new<D>(delegate: &D) -> Self
    where
        D: SolverDelegate<Interner = I>,
    {
        Self { obligations: Default::default(), usable_in_snapshot: delegate.num_open_snapshots() }
    }

    fn assert_usable_in_snapshot<D>(&self, delegate: &D)
    where
        D: SolverDelegate<Interner = I>,
    {
        assert_eq!(self.usable_in_snapshot, delegate.num_open_snapshots());
    }

    pub fn register<D>(&mut self, delegate: &D, obligation: PredicateObligation<I>)
    where
        D: SolverDelegate<Interner = I>,
    {
        self.assert_usable_in_snapshot(delegate);
        if let Some(GoalEvaluation { certainty, stalled_on, .. }) =
            compute_goal_fast_path(delegate, obligation.as_goal(), obligation.cause.span())
        {
            // If we can take the fast path, do not add a successful goal to
            // the pending obligations. For `Certainty::Maybe`, retain the
            // precise `stalled_on` information for later re-evaluation.
            match certainty {
                Certainty::Yes => {}
                Certainty::Maybe(_) => {
                    self.obligations.register(obligation, stalled_on);
                }
            }
        } else {
            self.obligations.register(obligation, None);
        }
    }

    fn on_fulfillment_overflow<D>(&mut self, delegate: &D)
    where
        D: SolverDelegate<Interner = I>,
    {
        delegate.probe(|| {
            // IMPORTANT: we must not resolve any inference variables in the
            // obligations, as this is all happening inside of a probe. The
            // probe makes sure we collect every obligation involved in the
            // overflow. Conceptually, we check which goals would change if we
            // performed one more fulfillment iteration.
            let overflowed = self
                .obligations
                .pending
                .extract_if(.., |(obligation, stalled_on)| {
                    let result = delegate.evaluate_root_goal(
                        obligation.as_goal(),
                        obligation.cause.span(),
                        stalled_on.take(),
                    );

                    matches!(result, Ok(GoalEvaluation { has_changed: HasChanged::Yes, .. }))
                })
                .map(|(obligation, _)| obligation)
                .collect::<Vec<_>>();

            self.obligations.overflowed.extend(overflowed);
        });
    }

    pub fn try_evaluate_obligations<D, Inspect, OnSuccess>(
        &mut self,
        delegate: &D,
        mut inspect: Inspect,
        mut on_success: OnSuccess,
    ) -> ThinVec<NextSolverError<I>>
    where
        D: SolverDelegate<Interner = I>,
        Inspect: FnMut(
            &PredicateObligation<I>,
            Goal<I, I::Predicate>,
            &Result<GoalEvaluation<I>, NoSolution>,
        ),
        OnSuccess: FnMut(&PredicateObligation<I>),
    {
        self.assert_usable_in_snapshot(delegate);

        let mut errors = ThinVec::new();

        loop {
            let mut any_changed = false;
            let mut overflowed = false;

            self.obligations.pending.retain_mut(|(obligation, opt_stalled_on)| {
                if overflowed {
                    return false;
                }

                // Common case: still stalled; keep the obligation. This path is extremely hot in
                // some cases; there can be thousands of pending obligations.
                if let Some(stalled_on) = opt_stalled_on
                    && delegate.goal_remains_stalled(stalled_on)
                {
                    return true;
                }

                let goal = obligation.as_goal();
                let result = delegate.evaluate_root_goal(
                    goal,
                    obligation.cause.span(),
                    opt_stalled_on.take(),
                );

                inspect(obligation, goal, &result);

                let GoalEvaluation { goal, certainty, has_changed, stalled_on } = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(NextSolverError::TrueError(obligation.clone()));
                        return false;
                    }
                };

                // We resolved the goal in `evaluate_root_goal`; retain the eagerly resolved
                // predicate to avoid repeating this work in the next iteration.
                obligation.predicate = goal.predicate;

                if has_changed == HasChanged::Yes {
                    // Track the number of times this root goal resulted in inference progress.
                    let depth = obligation.recursion_depth + 1;
                    obligation.recursion_depth = depth;

                    if depth > delegate.cx().recursion_limit() {
                        // We cannot break out of `retain_mut`, so use a flag and handle
                        // fulfillment overflow after the iteration.
                        overflowed = true;
                        return false;
                    }

                    any_changed = true;
                }

                match certainty {
                    Certainty::Yes => {
                        on_success(obligation);
                        false
                    }
                    Certainty::Maybe(_) => {
                        *opt_stalled_on = stalled_on;
                        true
                    }
                }
            });

            if overflowed {
                self.on_fulfillment_overflow(delegate);
                // Only return true errors accumulated while processing.
                return errors;
            }

            if !any_changed {
                break;
            }
        }

        errors
    }

    pub fn has_pending_obligations(&self) -> bool {
        self.obligations.has_pending_obligations()
    }

    pub fn pending_obligations(&self) -> ThinVec<PredicateObligation<I>> {
        self.obligations.clone_pending()
    }

    pub fn pending_obligations_filtered<F>(&self, filter: F) -> ThinVec<PredicateObligation<I>>
    where
        F: FnMut(&PredicateObligation<I>, &Option<GoalStalledOn<I>>) -> bool,
    {
        self.obligations.clone_pending_filtered(filter)
    }

    pub fn drain_pending_obligations<F>(&mut self, filter: F) -> ThinVec<PredicateObligation<I>>
    where
        F: FnMut(&PredicateObligation<I>, &Option<GoalStalledOn<I>>) -> bool,
    {
        self.obligations.drain_pending(filter)
    }

    pub fn collect_remaining_errors<E>(
        &mut self,
        map: impl FnMut(NextSolverError<I>) -> E,
    ) -> ThinVec<E> {
        self.obligations.collect_remaining_errors(map)
    }
}
