//! Routines to check for relations between fully inferred types.
//!
//! FIXME: Move this to a more general place. The utility of this extends to
//! other areas of the compiler as well.

use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt, Variance};
use rustc_trait_selection::traits::ObligationCtxt;

/// Returns whether the two types are equal up to subtyping.
///
/// This is used in case we don't know the expected subtyping direction
/// and still want to check whether anything is broken.
pub fn is_equal_up_to_subtyping<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    // Fast path.
    if src == dest {
        return true;
    }

    // Check for subtyping in either direction.
    relate_types(tcx, param_env, Variance::Covariant, src, dest)
        || relate_types(tcx, param_env, Variance::Covariant, dest, src)
}

/// Returns whether `src` is a subtype of `dest`, i.e. `src <: dest`.
///
/// When validating assignments, the variance should be `Covariant`. When checking
/// during `MirPhase` >= `MirPhase::Runtime(RuntimePhase::Initial)` variance should be `Invariant`
/// because we want to check for type equality.
pub fn relate_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    variance: Variance,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    if src == dest {
        return true;
    }

    let mut builder = tcx.infer_ctxt().ignoring_regions();
    let infcx = builder.build();
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
