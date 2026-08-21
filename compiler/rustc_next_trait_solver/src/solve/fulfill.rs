use std::marker::PhantomData;
use std::mem;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::{
    FromSolverError, PredicateObligation, PredicateObligations, TraitEngine, TraitErrors,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_next_trait_solver::solve::fast_path::compute_goal_fast_path;
use rustc_next_trait_solver::solve::{
    GoalEvaluation, GoalStalledOn, HasChanged, SolverDelegateEvalExt as _, StalledOnCoroutines,
};
use thin_vec::ThinVec;
use tracing::instrument;

use self::derive_errors::*;
use super::Certainty;
use super::delegate::SolverDelegate;
use crate::traits::{FulfillmentError, ScrubbedTraitError};

mod derive_errors;

// `ThinVec` is important for performance, but not for the usual memory layout reasons.
// `try_evaluate_obligations` is extremely hot and uses `retain_mut`. `ThinVec::retain_mut` is
// simple and sub-optimal in terms of how it moves elements, but it can be inlined.
// `Vec::retain_mut` is more sophisticated and minimizes element moves, but also contains more code
// and doesn't get inlined in `try_evaluate_obligations`, giving worse performance overall.
type PendingObligations<'tcx> =
    ThinVec<(PredicateObligation<'tcx>, Option<GoalStalledOn<TyCtxt<'tcx>>>)>;

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
    /// to avoid recomputing them each time `try_evaluate_obligations` is called.
    /// This also allows us to return the correct `FulfillmentError` for them.
    overflowed: Vec<PredicateObligation<'tcx>>,
    pending: PendingObligations<'tcx>,
}

impl<'tcx> ObligationStorage<'tcx> {
    fn register(
        &mut self,
        obligation: PredicateObligation<'tcx>,
        stalled_on: Option<GoalStalledOn<TyCtxt<'tcx>>>,
    ) {
        self.pending.push((obligation, stalled_on));
    }

    fn has_pending_obligations(&self) -> bool {
        !self.pending.is_empty() || !self.overflowed.is_empty()
    }

    fn clone_pending(&self) -> PredicateObligations<'tcx> {
        let mut obligations: PredicateObligations<'tcx> =
            self.pending.iter().map(|(o, _)| o.clone()).collect();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn clone_pending_filtered<F>(&self, f: F) -> PredicateObligations<'tcx>
    where
        F: FnMut(&&(PredicateObligation<'tcx>, Option<GoalStalledOn<TyCtxt<'tcx>>>)) -> bool,
    {
        let mut obligations: PredicateObligations<'tcx> =
            self.pending.iter().filter(f).map(|(o, _)| o.clone()).collect();
        obligations.extend(self.overflowed.iter().cloned());
        obligations
    }

    fn drain_pending(
        &mut self,
        cond: impl Fn(&PredicateObligation<'tcx>, &Option<GoalStalledOn<TyCtxt<'tcx>>>) -> bool,
    ) -> PendingObligations<'tcx> {
        let (unstalled, pending) =
            mem::take(&mut self.pending).into_iter().partition(|(o, s)| cond(o, s));
        self.pending = pending;
        unstalled
    }

    fn on_fulfillment_overflow(&mut self, infcx: &InferCtxt<'tcx>) {
        infcx.probe(|_| {
            // IMPORTANT: we must not use solve any inference variables in the obligations
            // as this is all happening inside of a probe. We use a probe to make sure
            // we get all obligations involved in the overflow. We pretty much check: if
            // we were to do another step of `try_evaluate_obligations`, which goals would
            // change.
            self.overflowed.extend(
                self.pending
                    .extract_if(.., |(o, stalled_on)| {
                        let goal = o.as_goal();
                        let result = <&SolverDelegate<'tcx>>::from(infcx).evaluate_root_goal(
                            goal,
                            o.cause.span,
                            stalled_on.take(),
                        );
                        matches!(result, Ok(GoalEvaluation { has_changed: HasChanged::Yes, .. }))
                    })
                    .map(|(o, _)| o),
            );
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
        infcx: &InferCtxt<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        result: &Result<GoalEvaluation<TyCtxt<'tcx>>, NoSolution>,
    ) {
        if let Some(inspector) = infcx.obligation_inspector.get() {
            let result = match result {
                Ok(GoalEvaluation { certainty, .. }) => Ok(*certainty),
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

        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        if let Some(GoalEvaluation { goal: _, certainty, has_changed: _, stalled_on }) =
            compute_goal_fast_path(delegate, obligation.as_goal(), obligation.cause.span)
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

    #[inline]
    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E> {
        if self.obligations.pending.is_empty() && self.obligations.overflowed.is_empty() {
            // Typically in more than 99.9% of cases this condition is true, therefore we outline
            // the other case.
            TraitErrors::NoErrors
        } else {
            TraitErrors::HasErrors(collect_remaining_errors_impl(self, infcx))
        }
    }

    fn try_evaluate_obligations(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E> {
        assert_eq!(self.usable_in_snapshot, infcx.num_open_snapshots());
        let mut errors = TraitErrors::NoErrors;
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
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

                let result = delegate.evaluate_root_goal(
                    obligation.as_goal(),
                    obligation.cause.span,
                    opt_stalled_on.take(),
                );
                Self::inspect_evaluated_obligation(infcx, &obligation, &result);
                let GoalEvaluation { goal, certainty, has_changed, stalled_on } = match result {
                    Ok(result) => result,
                    Err(NoSolution) => {
                        errors.push(E::from_solver_error(
                            infcx,
                            NextSolverError::TrueError(obligation.clone()),
                        ));
                        return false;
                    }
                };

                // We've resolved the goal in `evaluate_root_goal`, avoid redoing this work
                // in the next iteration. This does not resolve the inference variables
                // constrained by evaluating the goal.
                obligation.predicate = goal.predicate;
                if has_changed == HasChanged::Yes {
                    // We increment the recursion depth here to track the number of times
                    // this goal has resulted in inference progress. This doesn't precisely
                    // model the way that we track recursion depth in the old solver due
                    // to the fact that we only process root obligations, but it is a good
                    // approximation and should only result in fulfillment overflow in
                    // pathological cases.
                    obligation.recursion_depth += 1;

                    if !infcx.tcx.recursion_limit().value_within_limit(obligation.recursion_depth) {
                        // At this point we want to stop evaluating goals. We can't break out of
                        // `retain_mut`, so instead we set this flag which causes all other
                        // elements to be skipped.
                        overflowed = true;
                        return false;
                    } else {
                        any_changed = true;
                    }
                }

                match certainty {
                    Certainty::Yes => {
                        // Goals may depend on structural identity. Region uniquification at the
                        // start of MIR borrowck may cause things to no longer be so, potentially
                        // causing an ICE.
                        //
                        // While we uniquify root goals in HIR this does not handle cases where
                        // regions are hidden inside of a type or const inference variable.
                        //
                        // FIXME(-Znext-solver): This does not handle inference variables hidden
                        // inside of an opaque type, e.g. if there's `Opaque = (?x, ?x)` in the
                        // storage, we can also rely on structural identity of `?x` even if we
                        // later uniquify it in MIR borrowck.
                        if infcx.in_hir_typeck
                            && (obligation.has_non_region_infer() || obligation.has_free_regions())
                        {
                            infcx.push_hir_typeck_potentially_region_dependent_goal(
                                obligation.clone(),
                            );
                        }
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
            if overflowed {
                self.obligations.on_fulfillment_overflow(infcx);
                // Only return true errors that we have accumulated while processing.
                return errors;
            }

            if !any_changed {
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

    fn pending_obligations_potentially_referencing_sub_root(
        &self,
        infcx: &InferCtxt<'tcx>,
        vid: ty::TyVid,
    ) -> PredicateObligations<'tcx> {
        // `-Zdisable-fast-paths`: same gate as the other new-solver fast paths.
        if infcx.tcx.disable_trait_solver_fast_paths() {
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
            stalled_on.stalled_vars.iter().filter_map(|arg| arg.as_type(infcx.tcx)).any(|ty| {
                match *infcx.shallow_resolve(ty).kind() {
                    ty::Infer(ty::TyVar(tv)) => infcx.sub_unification_table_root_var(tv) == vid,
                    _ => true,
                }
            })
        })
    }

    fn pending_obligations_potentially_referencing_float_infer(
        &self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        // `-Zdisable-fast-paths`: same gate as the other new-solver fast paths.
        if infcx.tcx.disable_trait_solver_fast_paths() {
            return self.obligations.clone_pending();
        }

        self.obligations.clone_pending_filtered(|(_, stalled_on)| {
            let Some(stalled_on) = stalled_on else { return true };
            // If the stalled vars don't have float infers, the nested goals won't
            // have them either. We only create float infers for user written literals.
            stalled_on
                .stalled_vars
                .iter()
                .filter_map(|arg| arg.as_type(infcx.tcx))
                .any(|ty| matches!(infcx.shallow_resolve(ty).kind(), ty::Infer(ty::FloatVar(_))))
        })
    }

    fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        let stalled_coroutines = match infcx.typing_mode_raw().assert_not_erased() {
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
            .map(|(o, _)| o)
            .collect()
    }
}

#[cold]
#[inline(never)]
fn collect_remaining_errors_impl<'tcx, E>(
    cx: &mut FulfillmentCtxt<'tcx, E>,
    infcx: &InferCtxt<'tcx>,
) -> ThinVec<E>
where
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    cx.obligations
        .pending
        .drain(..)
        .map(|(obligation, _)| NextSolverError::Ambiguity(obligation))
        .chain(
            cx.obligations
                .overflowed
                .drain(..)
                .map(|obligation| NextSolverError::Overflow(obligation)),
        )
        .map(|e| E::from_solver_error(infcx, e))
        .collect()
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

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    // Before #160005 this pair was greater than 128 bytes, which triggered the use of (slow)
    // `memcpy` for moving elements of `PendingObligations`. Then #160479 greatly reduced the
    // number of `memcpy` operations in `try_evaluate_obligations`. So the size of this pair is
    // much less important than it was, but still shouldn't be changed without some thought.
    static_assert_size!((PredicateObligation<'_>, Option<GoalStalledOn<TyCtxt<'_>>>), 104);
    // tidy-alphabetical-end
}
