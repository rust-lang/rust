//! Routines to check for relations between fully inferred types.
//!
//! FIXME: Move this to a more general place. The utility of this extends to
//! other areas of the compiler as well.

use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{Ty, TyCtxt, TypingEnv, Variance};
use rustc_trait_selection::traits::ObligationCtxt;

/// Returns whether `src` is a subtype of `dest`, i.e. `src <: dest`.
pub fn sub_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    relate_types(tcx, typing_env, Variance::Covariant, src, dest)
}

/// Returns whether `src` is a subtype of `dest`, i.e. `src <: dest`.
///
/// When validating assignments, the variance should be `Covariant`. When checking
/// during `MirPhase` >= `MirPhase::Runtime(RuntimePhase::Initial)` variance should be `Invariant`
/// because we want to check for type equality.
pub fn relate_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    variance: Variance,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    if src == dest {
        return true;
    }

    let (infcx, param_env) = tcx.infer_ctxt().ignoring_regions().build_with_typing_env(typing_env);
    let ocx = ObligationCtxt::new(&infcx);
    let cause = ObligationCause::dummy();
    let src = ocx.normalize(&cause, param_env, src);
    let dest = ocx.normalize(&cause, param_env, dest);
    match ocx.relate(&cause, param_env, variance, src, dest) {
        Ok(()) => {}
        Err(_) => return false,
    };
    ocx.select_all_or_error().is_empty()
}
