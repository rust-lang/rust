use rustc_hir as hir;
use rustc_middle::ty::{self, CanonicalUserType, TyCtxt, UserType};

crate trait UserAnnotatedTyHelpers<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx>;

    fn typeck_results(&self) -> &ty::TypeckResults<'tcx>;

    /// Looks up the type associated with this hir-id and applies the
    /// user-given substitutions; the hir-id must map to a suitable
    /// type.
    fn user_substs_applied_to_ty_of_hir_id(
        &self,
        hir_id: hir::HirId,
    ) -> Option<CanonicalUserType<'tcx>> {
        let user_provided_types = self.typeck_results().user_provided_types();
        let mut user_ty = *user_provided_types.get(hir_id)?;
        debug!("user_subts_applied_to_ty_of_hir_id: user_ty={:?}", user_ty);
        let ty = self.typeck_results().node_type(hir_id);
        match ty.kind() {
            ty::Adt(adt_def, ..) => {
                if let UserType::TypeOf(ref mut did, _) = &mut user_ty.value {
                    *did = adt_def.did;
                }
                Some(user_ty)
            }
            ty::FnDef(..) => Some(user_ty),
            _ => bug!("ty: {:?} should not have user provided type {:?} recorded ", ty, user_ty),
        }
    }
}
