use std::fmt::Debug;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyVid, Upcast};
use thin_vec::{ThinVec, thin_vec};

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
    /// An old-solver-style cycle error, which will fatal. This is not
    /// returned by the new solver.
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

impl<'tcx> EngineError<'tcx> for ScrubbedTraitError<'tcx> {
    fn try_report_errors(_infcx: &InferCtxt<'tcx>, _errors: ThinVec<Self>) {}
}

#[derive(Debug, Clone)]
#[must_use]
pub enum TraitErrors<E> {
    HasErrors(ThinVec<E>),
    NoErrors,
}

impl<E> TraitErrors<E> {
    #[inline]
    pub fn from_iter(iter: impl ExactSizeIterator<Item = E>) -> TraitErrors<E> {
        if iter.len() == 0 { TraitErrors::NoErrors } else { TraitErrors::HasErrors(iter.collect()) }
    }

    #[inline]
    pub fn has_errors(&self) -> bool {
        matches!(self, TraitErrors::HasErrors(_))
    }

    #[inline]
    pub fn no_errors(&self) -> bool {
        matches!(self, TraitErrors::NoErrors)
    }

    #[inline]
    pub fn as_slice(&self) -> &[E] {
        match self {
            TraitErrors::HasErrors(errors) => errors.as_slice(),
            TraitErrors::NoErrors => &[],
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [E] {
        match self {
            TraitErrors::HasErrors(errors) => errors.as_mut_slice(),
            TraitErrors::NoErrors => &mut [],
        }
    }

    #[inline]
    pub fn into_thin_vec(self) -> ThinVec<E> {
        match self {
            TraitErrors::HasErrors(errors) => errors,
            TraitErrors::NoErrors => ThinVec::new(),
        }
    }

    #[cold]
    pub fn push(&mut self, err: E) {
        match self {
            TraitErrors::HasErrors(errors) => errors.push(err),
            TraitErrors::NoErrors => *self = TraitErrors::HasErrors(thin_vec![err]),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TraitErrors::HasErrors(errors) => errors.len(),
            TraitErrors::NoErrors => 0,
        }
    }
}

impl<E> IntoIterator for TraitErrors<E> {
    type Item = E;
    type IntoIter = thin_vec::IntoIter<E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_thin_vec().into_iter()
    }
}

impl<'a, E> IntoIterator for &'a TraitErrors<E> {
    type Item = &'a E;
    type IntoIter = std::slice::Iter<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
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

    /// Go over the list of pending obligations and try to evaluate them.
    ///
    /// For each result:
    /// Ok: remove the obligation from the list
    /// Ambiguous: leave the obligation in the list to be evaluated later
    /// Err: remove the obligation from the list and return an error
    ///
    /// Returns a list of errors from obligations that evaluated to Err.
    #[must_use]
    fn try_evaluate_obligations(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E>;

    fn collect_remaining_errors(&mut self, infcx: &InferCtxt<'tcx>) -> TraitErrors<E>;

    /// Evaluate all pending obligations, return error if they can't be evaluated.
    ///
    /// For each result:
    /// Ok: remove the obligation from the list
    /// Ambiguous: remove the obligation from the list and return an error
    /// Err: remove the obligation from the list and return an error
    ///
    /// Returns a list of errors from obligations that evaluated to Ambiguous or Err.
    #[must_use]
    fn evaluate_obligations_error_on_ambiguity(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> TraitErrors<E> {
        let errors = self.try_evaluate_obligations(infcx);
        if errors.has_errors() {
            return errors;
        }

        self.collect_remaining_errors(infcx)
    }

    fn has_pending_obligations(&self) -> bool;

    fn pending_obligations(&self) -> PredicateObligations<'tcx>;

    /// Pending obligations potentially referencing an inference variable whose
    /// sub-unification root is `_sub_root`. May be conservative: implementations
    /// can return obligations that don't actually reference `_sub_root` (the
    /// default just returns everything).
    fn pending_obligations_potentially_referencing_sub_root(
        &self,
        _infcx: &InferCtxt<'tcx>,
        _sub_root: TyVid,
    ) -> PredicateObligations<'tcx> {
        self.pending_obligations()
    }

    /// Pending obligations potentially referencing float inference variables.
    ///
    /// FIXME: use a generic filter for `pending_obligations_potentially_referencing_sub_root`
    /// and this after `TraitEngine` doesn't need to be dyn compatible.
    fn pending_obligations_potentially_referencing_float_infer(
        &self,
        _infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx> {
        self.pending_obligations()
    }

    /// Among all pending obligations, collect those are stalled on a inference variable which has
    /// changed since the last call to `try_evaluate_obligations`. Those obligations are marked as
    /// successful and returned.
    fn drain_stalled_obligations_for_coroutines(
        &mut self,
        infcx: &InferCtxt<'tcx>,
    ) -> PredicateObligations<'tcx>;
}

pub trait EngineError<'tcx>: Sized + 'tcx {
    fn try_report_errors(infcx: &InferCtxt<'tcx>, errors: ThinVec<Self>);
}

pub trait FromSolverError<'tcx, E>: EngineError<'tcx> + Debug {
    fn from_solver_error(infcx: &InferCtxt<'tcx>, error: E) -> Self;
}
