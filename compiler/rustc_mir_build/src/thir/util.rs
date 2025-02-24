use rustc_hir as hir;
use rustc_middle::bug;
use rustc_middle::ty::{self, CanonicalUserType};
use tracing::debug;

/// Looks up the type associated with this hir-id and applies the
/// user-given generic parameters; the hir-id must map to a suitable
/// type.
pub(crate) fn user_args_applied_to_ty_of_hir_id<'tcx>(
    typeck_results: &ty::TypeckResults<'tcx>,
    hir_id: hir::HirId,
) -> Option<CanonicalUserType<'tcx>> {
    let user_provided_types = typeck_results.user_provided_types();
    let mut user_ty = *user_provided_types.get(hir_id)?;
    debug!("user_subts_applied_to_ty_of_hir_id: user_ty={:?}", user_ty);
    let ty = typeck_results.node_type(hir_id);
    match ty.kind() {
        ty::Adt(adt_def, ..) => {
            if let ty::UserTypeKind::TypeOf(did, _) = &mut user_ty.value.kind {
                *did = adt_def.did();
            }
            Some(user_ty)
        }
        ty::FnDef(..) => Some(user_ty),
        _ => bug!("ty: {:?} should not have user provided type {:?} recorded ", ty, user_ty),
    }
}
