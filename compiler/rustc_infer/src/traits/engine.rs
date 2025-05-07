use std::fmt::Debug;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, Upcast};

use super::{ObligationCause, PredicateObligation, PredicateObligations};
use crate::infer::InferCtxt;
use crate::traits::Obligation;

/// A trait error with most of its information removed. This is the error
/// returned by an `ObligationCtxt` by default, and suitable if you just
/// want to see if a predicate holds, and don't particularly care about the
/// error itself (except for if it's an ambiguity or true error).
///
/// use `ObligationCtxt::new_with_diagnostics` to get a `FulfillmentError`.
#[derive(Clone, Debug)]
pub enum ScrubbedTraitError<'tcx> {
    /// A real error. This goal definitely does not hold.
    TrueError,
    /// An ambiguity. This goal may hold if further inference is done.
    Ambiguity,
    /// An old-solver-style cycle error, which will fatal.
    Cycle(PredicateObligations<'tcx>),
}

impl<'tcx> ScrubbedTraitError<'tcx> {
    pub fn is_true_error(&self) -> bool {
        match self {
            ScrubbedTraitError::TrueError => true,
            ScrubbedTraitError::Ambiguity | ScrubbedTraitError::Cycle(_) => false,
        }
    }
}

pub trait TraitEngine<'tcx, E: 'tcx>: 'tcx {
    /// Requires that `ty` must implement the trait with `def_id` in
    /// the given environment. This trait must not have any type
    /// parameters (except for `Self`).
    fn register_bound(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        def_id: DefId,
        cause: ObligationCause<'tcx>,
    ) {
        let trait_ref = ty::TraitRef::new(infcx.tcx, def_id, [ty]);
        self.register_predicate_obligation(
            infcx,
            Obligation {
                cause,
                recursion_depth: 0,
                param_env,
                predicate: trait_ref.upcast(infcx.tcx),
            },
        );
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    );

    fn register_predicate_obligations(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        obligations: PredicateObligations<'tcx>,
    ) {
        for obligation in obligations {
            self.register_predicate_obligation(infcx, obligation);
        }
    }

    #[must_use]
    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E>;

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E>;

    #[must_use]
    fn select_all_or_error(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<E> {
        let errors = self.select_where_possible(infcx);
        if !errors.is_empty() {
            return errors;
        }

        self.collect_remaining_errors(infcx)
    }

    fn has_pending_obligations(&self) -> bool;

    fn pending_obligations(&self) -> PredicateObligations<'tcx>;

    /// Among all pending obligations, collect those are stalled on a inference variable which has
    /// changed since the last call to `select_where_possible`. Those obligations are marked as
    /// successful and returned.
    fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx>;
}

pub trait FromSolverError<'tcx, E>: Debug + 'tcx {
    fn from_solver_error(infcx: &InferCtxt<'tcx>, error: E) -> Self;
}
