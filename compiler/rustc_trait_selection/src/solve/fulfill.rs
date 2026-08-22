use std::marker::PhantomData;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::{
    FromSolverError, PredicateObligation, PredicateObligations, TraitEngine, TraitErrors,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_next_trait_solver::solve::{
    FulfillmentCtxt as SolverFulfillmentCtxt, GoalEvaluation,
    NextSolverError as SolverNextSolverError, StalledOnCoroutines,
};
use tracing::instrument;

use self::derive_errors::*;
use super::delegate::SolverDelegate;
use crate::traits::{FulfillmentError, ScrubbedTraitError};

mod derive_errors;

/// A trait engine using the new trait solver.
///
/// The frontend wrapper keeps rustc-specific diagnostics and
/// successful-obligation handling outside of the shared engine.
pub struct FulfillmentCtxt<'tcx, E: 'tcx> {
    core: SolverFulfillmentCtxt<TyCtxt<'tcx>>,
    _errors: PhantomData<E>,
}

impl<'tcx, E: 'tcx> FulfillmentCtxt<'tcx, E> {
    pub fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentCtxt<'tcx, E> {
        assert!(
            infcx.next_trait_solver(),
            "new trait solver fulfillment context created when \
            infcx is set up for old trait solver"
        );
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        FulfillmentCtxt { core: SolverFulfillmentCtxt::new(delegate), _errors: PhantomData }
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

impl<'tcx> From<SolverNextSolverError<TyCtxt<'tcx>>> for NextSolverError<'tcx> {
    fn from(error: SolverNextSolverError<TyCtxt<'tcx>>) -> Self {
        match error {
            SolverNextSolverError::TrueError(obligation) => Self::TrueError(obligation),
            SolverNextSolverError::Ambiguity(obligation) => Self::Ambiguity(obligation),
            SolverNextSolverError::Overflow(obligation) => Self::Overflow(obligation),
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
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        self.core.register(delegate, obligation);
    }

    #[inline]
    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E> {
        if !self.core.has_pending_obligations() {
            TraitErrors::NoErrors
        } else {
            TraitErrors::HasErrors(
                self.core
                    .collect_remaining_errors(|error| E::from_solver_error(infcx, error.into())),
            )
        }
    }

    fn try_evaluate_obligations(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E> {
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);

        let errors = self.core.try_evaluate_obligations(
            delegate,
            |obligation, _, result| {
                Self::inspect_evaluated_obligation(infcx, obligation, result);
            },
            |obligation| {
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
                    infcx.push_hir_typeck_potentially_region_dependent_goal(obligation.clone());
                }
            },
        );

        TraitErrors::from_iter(
            errors
                .into_iter()
                .map(NextSolverError::from)
                .map(|error| E::from_solver_error(infcx, error)),
        )
    }

    fn has_pending_obligations(&self) -> bool {
        self.core.has_pending_obligations()
    }

    fn pending_obligations(&self) -> PredicateObligations<'tcx> {
        self.core.pending_obligations()
    }

    fn pending_obligations_potentially_referencing_sub_root(
        &self,
        infcx: &InferCtxt<'tcx>,
        vid: ty::TyVid,
    ) -> PredicateObligations<'tcx> {
        // `-Zdisable-fast-paths`: same gate as the other new-solver fast paths.
        if infcx.tcx.disable_trait_solver_fast_paths() {
            return self.pending_obligations();
        }

        self.core.pending_obligations_filtered(|_, stalled_on| {
            let Some(stalled_on) = stalled_on else {
                return true;
            };

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
            return self.pending_obligations();
        }

        self.core.pending_obligations_filtered(|_, stalled_on| {
            let Some(stalled_on) = stalled_on else {
                return true;
            };

            // If the stalled vars don't have float infers, the nested goals
            // won't have them either. We only create float infers for
            // user-written literals.
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

        self.core.drain_pending_obligations(|_, stalled_on| {
            stalled_on.as_ref().is_some_and(|stalled_on| {
                matches!(
                    stalled_on.stalled_maybe_info.stalled_on_coroutines,
                    StalledOnCoroutines::Yes
                )
            })
        })
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

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    use crate::solve::GoalStalledOn;

    // tidy-alphabetical-start
    // Before #160005 this pair was greater than 128 bytes, which triggered the use of (slow)
    // `memcpy` for moving elements of `PendingObligations`.
    static_assert_size!((PredicateObligation<'_>, Option<GoalStalledOn<TyCtxt<'_>>>), 104);
    // tidy-alphabetical-end
}
