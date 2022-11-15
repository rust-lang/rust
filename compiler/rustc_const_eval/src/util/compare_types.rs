//! Routines to check for relations between fully inferred types.
//!
//! FIXME: Move this to a more general place. The utility of this extends to
//! other areas of the compiler as well.

use rustc_infer::infer::{DefiningAnchor, TyCtxtInferExt};
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt};
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
    is_subtype(tcx, param_env, src, dest) || is_subtype(tcx, param_env, dest, src)
}

/// Returns whether `src` is a subtype of `dest`, i.e. `src <: dest`.
///
/// This mostly ignores opaque types as it can be used in constraining contexts
/// while still computing the final underlying type.
pub fn is_subtype<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    if src == dest {
        return true;
    }

    let mut builder =
        tcx.infer_ctxt().ignoring_regions().with_opaque_type_inference(DefiningAnchor::Bubble);
    let infcx = builder.build();
    let ocx = ObligationCtxt::new(&infcx);
    let cause = ObligationCause::dummy();
    let src = ocx.normalize(cause.clone(), param_env, src);
    let dest = ocx.normalize(cause.clone(), param_env, dest);
    let Ok(infer_ok) = infcx.at(&cause, param_env).sub(src, dest) else {
        return false;
    };
    let () = ocx.register_infer_ok_obligations(infer_ok);
    let errors = ocx.select_all_or_error();
    // With `Reveal::All`, opaque types get normalized away, with `Reveal::UserFacing`
    // we would get unification errors because we're unable to look into opaque types,
    // even if they're constrained in our current function.
    //
    // It seems very unlikely that this hides any bugs.
    let _ = infcx.inner.borrow_mut().opaque_type_storage.take_opaque_types();
    errors.is_empty()
}
