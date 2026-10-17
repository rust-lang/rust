use std::marker::PhantomData;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::{
    FromSolverError, PredicateObligation, PredicateObligations, TraitEngine, TraitErrors,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_next_trait_solver::solve::{FulfillmentCtxt as SolverFulfillmentCtxt, GoalEvaluation};
use thin_vec::ThinVec;
use tracing::instrument;

use self::derive_errors::*;
use super::delegate::SolverDelegate;
use crate::error_reporting::InferCtxtErrorExt;
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
    core: SolverFulfillmentCtxt<TyCtxt<'tcx>>,
    _errors: PhantomData<E>,
}

impl<'tcx, E: 'tcx> FulfillmentCtxt<'tcx, E> {
    pub fn new(infcx: &InferCtxt<'tcx>) -> FulfillmentCtxt<'tcx, E> {
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        FulfillmentCtxt {
            core: SolverFulfillmentCtxt::new(delegate, infcx.num_open_snapshots()),
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
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        self.core.register(delegate, infcx.num_open_snapshots(), obligation);
    }

    #[inline]
    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E> {
        if !self.core.has_pending_obligations() {
            // Typically in more than 99.9% of cases this condition is true, therefore we outline
            // the other case.
            TraitErrors::NoErrors
        } else {
            TraitErrors::HasErrors(collect_remaining_errors_impl(self, infcx))
        }
    }

    fn try_evaluate_obligations(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E> {
        let mut errors = TraitErrors::NoErrors;
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);

        self.core.try_evaluate_obligations(
            delegate,
            infcx.num_open_snapshots(),
            |obligation, result| {
                Self::inspect_evaluated_obligation(infcx, obligation, result);
            },
            |obligation| {
                errors.push(E::from_solver_error(infcx, NextSolverError::TrueError(obligation)));
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
            |obligation| infcx.err_ctxt().report_overflow_obligation(obligation, true),
        );

        errors
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
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        self.core.pending_obligations_potentially_referencing_sub_root(delegate, vid)
    }

    fn pending_obligations_potentially_referencing_float_infer(
        &self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        self.core.pending_obligations_potentially_referencing_float_infer(delegate)
    }

    fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        self.core.drain_stalled_obligations_for_coroutines(delegate)
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
    cx.core
        .drain_remaining_obligations()
        .map(NextSolverError::Ambiguity)
        .map(|e| E::from_solver_error(infcx, e))
        .collect()
}

pub enum NextSolverError<'tcx> {
    TrueError(PredicateObligation<'tcx>),
    Ambiguity(PredicateObligation<'tcx>),
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
        }
    }
}

impl<'tcx> FromSolverError<'tcx, NextSolverError<'tcx>> for ScrubbedTraitError<'tcx> {
    fn from_solver_error(_infcx: &InferCtxt<'tcx>, error: NextSolverError<'tcx>) -> Self {
        match error {
            NextSolverError::TrueError(_) => ScrubbedTraitError::TrueError,
            NextSolverError::Ambiguity(_) => ScrubbedTraitError::Ambiguity,
        }
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;
    use rustc_next_trait_solver::solve::GoalStalledOn;

    use super::*;
    // tidy-alphabetical-start
    // Before #160005 this pair was greater than 128 bytes, which triggered the use of (slow)
    // `memcpy` for moving elements of `PendingObligations`. Then #160479 greatly reduced the
    // number of `memcpy` operations in `try_evaluate_obligations`. So the size of this pair is
    // much less important than it was, but still shouldn't be changed without some thought.
    static_assert_size!((PredicateObligation<'_>, Option<GoalStalledOn<TyCtxt<'_>>>), 104);
    // tidy-alphabetical-end
}
