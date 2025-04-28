use std::assert_matches::assert_matches;

use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_middle::ty::{self, CanonicalUserType, TyCtxt};
use tracing::debug;

/// Looks up the type associated with this hir-id and applies the
/// user-given generic parameters; the hir-id must map to a suitable
/// type.
pub(crate) fn user_args_applied_to_ty_of_hir_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    typeck_results: &ty::TypeckResults<'tcx>,
    hir_id: hir::HirId,
) -> Option<CanonicalUserType<'tcx>> {
    let user_provided_types = typeck_results.user_provided_types();
    let mut user_ty = *user_provided_types.get(hir_id)?;
    debug!("user_subts_applied_to_ty_of_hir_id: user_ty={:?}", user_ty);
    let ty = typeck_results.node_type(hir_id);
    match ty.kind() {
        ty::Adt(adt_def, ..) => {
            // This "fixes" user type annotations for tupled ctor patterns for ADTs.
            // That's because `type_of(ctor_did)` returns a FnDef, but we actually
            // want to be annotating the type of the ADT itself. It's a bit goofy,
            // but it's easier to adjust this here rather than in the path lowering
            // code for patterns in HIR.
            if let ty::UserTypeKind::TypeOf(did, _) = &mut user_ty.value.kind {
                // This is either already set up correctly (struct, union, enum, or variant),
                // or needs adjusting (ctor). Make sure we don't start adjusting other
                // user annotations like consts or fn calls.
                assert_matches!(
                    tcx.def_kind(*did),
                    DefKind::Ctor(..)
                        | DefKind::Struct
                        | DefKind::Enum
                        | DefKind::Union
                        | DefKind::Variant
                );
                *did = adt_def.did();
            }
            Some(user_ty)
        }
        ty::FnDef(..) => Some(user_ty),
        _ => bug!("ty: {:?} should not have user provided type {:?} recorded ", ty, user_ty),
    }
}
