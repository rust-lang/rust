use std::mem;

use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{NoSolution, Obligation};
use rustc_type_ir::{self as ty, InferCtxtLike as _, Interner, TypingMode};
use thin_vec::ThinVec;
use tracing::instrument;

use super::fast_path::compute_goal_fast_path;
use super::{
    Certainty, GoalEvaluation, GoalStalledOn, HasChanged, SolverDelegateEvalExt as _,
    StalledOnCoroutines,
};
use crate::delegate::SolverDelegate;

// `ThinVec` is important for performance, but not for the usual memory layout reasons.
// `try_evaluate_obligations` is extremely hot and uses `retain_mut`. `ThinVec::retain_mut` is
// simple and sub-optimal in terms of how it moves elements, but it can be inlined.
// `Vec::retain_mut` is more sophisticated and minimizes element moves, but also contains more code
// and doesn't get inlined in `try_evaluate_obligations`, giving worse performance overall.
type PredicateObligation<I> = Obligation<I, <I as Interner>::Predicate>;
type PredicateObligations<I> = ThinVec<PredicateObligation<I>>;
type PendingObligations<I> = ThinVec<(PredicateObligation<I>, Option<GoalStalledOn<I>>)>;

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
pub struct FulfillmentCtxt<I: Interner> {
    obligations: ObligationStorage<I>,

    /// The snapshot in which this context was created. Using the context
    /// outside of this snapshot leads to subtle bugs if the snapshot
    /// gets rolled back. Because of this we explicitly check that we only
    /// use the context in exactly this snapshot.
    usable_in_snapshot: usize,
}

#[derive(Debug)]
struct ObligationStorage<I: Interner> {
    pending: PendingObligations<I>,
}

impl<I: Interner> Default for ObligationStorage<I> {
    fn default() -> Self {
        Self { pending: ThinVec::new() }
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
        !self.pending.is_empty()
    }

    fn clone_pending(&self) -> PredicateObligations<I> {
        self.pending.iter().map(|(o, _)| o.clone()).collect()
    }

    fn clone_pending_filtered<F>(&self, f: F) -> PredicateObligations<I>
    where
        F: FnMut(&&(PredicateObligation<I>, Option<GoalStalledOn<I>>)) -> bool,
    {
        self.pending.iter().filter(f).map(|(o, _)| o.clone()).collect()
    }

    fn drain_pending(
        &mut self,
        cond: impl Fn(&PredicateObligation<I>, &Option<GoalStalledOn<I>>) -> bool,
    ) -> PendingObligations<I> {
        let (unstalled, pending) =
            mem::take(&mut self.pending).into_iter().partition(|(o, s)| cond(o, s));
        self.pending = pending;
        unstalled
    }
}

impl<I: Interner> FulfillmentCtxt<I> {
    pub fn new<D>(delegate: &D, usable_in_snapshot: usize) -> Self
    where
        D: SolverDelegate<Interner = I>,
    {
        assert!(
            delegate.next_trait_solver(),
            "new trait solver fulfillment context created when \
            infcx is set up for old trait solver"
        );
        FulfillmentCtxt { obligations: Default::default(), usable_in_snapshot }
    }

    #[instrument(level = "trace", skip(self, delegate))]
    pub fn register<D>(
        &mut self,
        delegate: &D,
        current_snapshot: usize,
        obligation: PredicateObligation<I>,
    ) where
        D: SolverDelegate<Interner = I>,
    {
        assert_eq!(self.usable_in_snapshot, current_snapshot);

        if let Some(GoalEvaluation { goal: _, certainty, has_changed: _, stalled_on }) =
            compute_goal_fast_path(delegate, obligation.as_goal(), obligation.cause.span())
        {
            // If we can take the fast path, don't even bother adding the goal to obligations,
            // or if `Certainty::Maybe`, add it with precise stalled_on information.
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

    pub fn try_evaluate_obligations<D, Inspect, OnError, OnSuccess, OnOverflow>(
        &mut self,
        delegate: &D,
        current_snapshot: usize,
        mut inspect: Inspect,
        mut on_error: OnError,
        mut on_success: OnSuccess,
        mut on_overflow: OnOverflow,
    ) where
        D: SolverDelegate<Interner = I>,
        Inspect: FnMut(&PredicateObligation<I>, &Result<GoalEvaluation<I>, NoSolution>),
        OnError: FnMut(PredicateObligation<I>),
        OnSuccess: FnMut(&PredicateObligation<I>),
        OnOverflow: FnMut(&PredicateObligation<I>),
    {
        assert_eq!(self.usable_in_snapshot, current_snapshot);
        loop {
            let mut any_changed = false;

            self.obligations.pending.retain_mut(|(obligation, opt_stalled_on)| {
                // Common case: still stalled; keep the obligation. This path is extremely hot in
                // some cases; there can be thousands of pending obligations.
                if let Some(stalled_on) = opt_stalled_on
                    && delegate.goal_remains_stalled(stalled_on)
                {
                    return true;
                }

                let result = delegate.evaluate_root_goal(
                    obligation.as_goal(),
                    obligation.cause.span(),
                    opt_stalled_on.take(),
                );
                inspect(obligation, &result);
                let GoalEvaluation { goal, certainty, has_changed, stalled_on } = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        on_error(obligation.clone());
                        return false;
                    }
                };

                // We've resolved the goal in `evaluate_root_goal`, avoid redoing this work
                // in the next iteration. This does not resolve the inference variables
                // constrained by evaluating the goal.
                obligation.predicate = goal.predicate;
                if has_changed == HasChanged::Yes {
                    if obligation.recursion_depth > delegate.cx().recursion_limit() {
                        // We limit the total count of inference progress to avoid hang so we don't
                        // try to recover from this.
                        // It's more complicated to collect all overflows thus we stopped doing that.
                        // Eager aborting is also what the old solver does.
                        //
                        // Note: it's incredibly rare to actually encounter fulfillment overflow
                        // as a single obligation would have to result in different inference progress
                        // a `recursion_depth` number of times. This mostly happens in bugs or with
                        // `Subtype` obligations because we no longer use the `sub_unification_table`
                        // in generalization.
                        on_overflow(obligation);
                        unreachable!("fulfillment overflow handler must not return");
                    } else {
                        // We increment the recursion depth here to track the number of times
                        // this goal has resulted in inference progress. This doesn't precisely
                        // model the way that we track recursion depth in the old solver due
                        // to the fact that we only process root obligations, but it is a good
                        // approximation and should only result in fulfillment overflow in
                        // pathological cases.
                        obligation.recursion_depth += 1;
                        any_changed = true;
                    }
                }

                match certainty {
                    Certainty::Yes => {
                        on_success(obligation);
                        false
                    }
                    Certainty::Maybe(_) => {
                        // Update `opt_stalled_on` goal, for the next retain_mut, because we are
                        // running until a fixpoint.
                        *opt_stalled_on = stalled_on;
                        true
                    }
                }
            });

            if !any_changed {
                break;
            }
        }
    }

    pub fn has_pending_obligations(&self) -> bool {
        self.obligations.has_pending_obligations()
    }

    pub fn pending_obligations(&self) -> PredicateObligations<I> {
        self.obligations.clone_pending()
    }

    pub fn pending_obligations_potentially_referencing_sub_root<D>(
        &self,
        delegate: &D,
        vid: ty::TyVid,
    ) -> PredicateObligations<I>
    where
        D: SolverDelegate<Interner = I>,
    {
        // `-Zdisable-fast-paths`: same gate as the other new-solver fast paths.
        if delegate.disable_trait_solver_fast_paths() {
            return self.obligations.clone_pending();
        }
        self.obligations.clone_pending_filtered(|(_, stalled_on)| {
            let Some(stalled_on) = stalled_on else { return true };
            // Don't reuse the sub-unification roots cached on `stalled_on`:
            // a later sub-unification merge can have changed which root
            // each stalled var belongs to, so the cached info can be stale.
            // Walk `stalled_vars` and recompute the current root instead.
            //
            // Conservative here: if a stalled var no longer resolves to an
            // infer var, some unification happened, so the goal is no longer
            // stalled. Include it to be re-evaluated downstream.
            stalled_on.stalled_vars.iter().filter_map(|arg| arg.as_type(delegate.cx())).any(|ty| {
                match delegate.shallow_resolve(ty).kind() {
                    ty::Infer(ty::TyVar(tv)) => delegate.sub_unification_table_root_var(tv) == vid,
                    _ => true,
                }
            })
        })
    }

    pub fn pending_obligations_potentially_referencing_float_infer<D>(
        &self,
        delegate: &D,
    ) -> PredicateObligations<I>
    where
        D: SolverDelegate<Interner = I>,
    {
        // `-Zdisable-fast-paths`: same gate as the other new-solver fast paths.
        if delegate.disable_trait_solver_fast_paths() {
            return self.obligations.clone_pending();
        }

        self.obligations.clone_pending_filtered(|(_, stalled_on)| {
            let Some(stalled_on) = stalled_on else { return true };
            // If the stalled vars don't have float infers, the nested goals won't
            // have them either. We only create float infers for user written literals.
            stalled_on
                .stalled_vars
                .iter()
                .filter_map(|arg| arg.as_type(delegate.cx()))
                .any(|ty| matches!(delegate.shallow_resolve(ty).kind(), ty::Infer(ty::FloatVar(_))))
        })
    }

    pub fn drain_stalled_obligations_for_coroutines<D>(
        &mut self,
        delegate: &D,
    ) -> PredicateObligations<I>
    where
        D: SolverDelegate<Interner = I>,
    {
        let stalled_coroutines = match delegate.typing_mode_raw().assert_not_erased() {
            TypingMode::Typeck { defining_opaque_types_and_generators } => {
                defining_opaque_types_and_generators
            }
            TypingMode::Coherence
            | TypingMode::PostTypeckUntilBorrowck { defining_opaque_types: _ }
            | TypingMode::PostBorrowck { defined_opaque_types: _ }
            | TypingMode::Reflection
            | TypingMode::PostAnalysis
            | TypingMode::Codegen => return Default::default(),
        };

        if stalled_coroutines.is_empty() {
            return Default::default();
        }

        self.obligations
            .drain_pending(|_, stalled_on| {
                stalled_on.as_ref().is_some_and(|s| {
                    match s.stalled_maybe_info.stalled_on_coroutines {
                        StalledOnCoroutines::Yes => true,
                        StalledOnCoroutines::No => false,
                    }
                })
            })
            .into_iter()
            .map(|(obligation, _)| obligation)
            .collect()
    }

    pub fn drain_remaining_obligations(
        &mut self,
    ) -> impl Iterator<Item = PredicateObligation<I>> + '_ {
        self.obligations.pending.drain(..).map(|(obligation, _)| obligation)
    }
}
