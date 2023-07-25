//! Routines to check for relations between fully inferred types.
//!
//! FIXME: Move this to a more general place. The utility of this extends to
//! other areas of the compiler as well.

use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::traits::{DefiningAnchor, ObligationCause};
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
    let src = ocx.normalize(&cause, param_env, src);
    let dest = ocx.normalize(&cause, param_env, dest);
    match ocx.sub(&cause, param_env, src, dest) {
        Ok(()) => {}
        Err(_) => return false,
    };
    let errors = ocx.select_all_or_error();
    // With `Reveal::All`, opaque types get normalized away, with `Reveal::UserFacing`
    // we would get unification errors because we're unable to look into opaque types,
    // even if they're constrained in our current function.
    for (key, ty) in infcx.take_opaque_types() {
        span_bug!(
            ty.hidden_type.span,
            "{}, {}",
            tcx.type_of(key.def_id).instantiate(tcx, key.args),
            ty.hidden_type.ty
        );
    }
    errors.is_empty()
}
