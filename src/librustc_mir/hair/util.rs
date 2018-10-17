// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::mir::UserTypeAnnotation;
use rustc::ty::{self, AdtDef, TyCtxt};

crate trait UserAnnotatedTyHelpers<'gcx: 'tcx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'_, 'gcx, 'tcx>;

    fn tables(&self) -> &ty::TypeckTables<'tcx>;

    fn user_substs_applied_to_adt(
        &self,
        hir_id: hir::HirId,
        adt_def: &'tcx AdtDef,
    ) -> Option<UserTypeAnnotation<'tcx>> {
        let user_substs = self.tables().user_substs(hir_id)?;
        Some(UserTypeAnnotation::TypeOf(adt_def.did, user_substs))
    }

    /// Looks up the type associated with this hir-id and applies the
    /// user-given substitutions; the hir-id must map to a suitable
    /// type.
    fn user_substs_applied_to_ty_of_hir_id(
        &self,
        hir_id: hir::HirId,
    ) -> Option<UserTypeAnnotation<'tcx>> {
        let user_substs = self.tables().user_substs(hir_id)?;
        match &self.tables().node_id_to_type(hir_id).sty {
            ty::Adt(adt_def, _) => Some(UserTypeAnnotation::TypeOf(adt_def.did, user_substs)),
            ty::FnDef(def_id, _) => Some(UserTypeAnnotation::TypeOf(*def_id, user_substs)),
            sty => bug!(
                "sty: {:?} should not have user-substs {:?} recorded ",
                sty,
                user_substs
            ),
        }
    }
}
