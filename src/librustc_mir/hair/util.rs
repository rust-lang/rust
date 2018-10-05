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
use rustc::ty::{self, AdtDef, CanonicalTy, TyCtxt};

crate trait UserAnnotatedTyHelpers<'gcx: 'tcx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'_, 'gcx, 'tcx>;

    fn tables(&self) -> &ty::TypeckTables<'tcx>;

    fn user_substs_applied_to_adt(
        &self,
        hir_id: hir::HirId,
        adt_def: &'tcx AdtDef,
    ) -> Option<CanonicalTy<'tcx>> {
        let user_substs = self.tables().user_substs(hir_id)?;
        Some(user_substs.unchecked_map(|user_substs| {
            // Here, we just pair an `AdtDef` with the
            // `user_substs`, so no new types etc are introduced.
            self.tcx().mk_adt(adt_def, user_substs)
        }))
    }

    /// Looks up the type associated with this hir-id and applies the
    /// user-given substitutions; the hir-id must map to a suitable
    /// type.
    fn user_substs_applied_to_ty_of_hir_id(&self, hir_id: hir::HirId) -> Option<CanonicalTy<'tcx>> {
        let user_substs = self.tables().user_substs(hir_id)?;
        match &self.tables().node_id_to_type(hir_id).sty {
            ty::Adt(adt_def, _) => Some(user_substs.unchecked_map(|user_substs| {
                // Ok to call `unchecked_map` because we just pair an
                // `AdtDef` with the `user_substs`, so no new types
                // etc are introduced.
                self.tcx().mk_adt(adt_def, user_substs)
            })),
            ty::FnDef(def_id, _) => Some(user_substs.unchecked_map(|user_substs| {
                // Here, we just pair a `DefId` with the
                // `user_substs`, so no new types etc are introduced.
                self.tcx().mk_fn_def(*def_id, user_substs)
            })),
            sty => bug!(
                "sty: {:?} should not have user-substs {:?} recorded ",
                sty,
                user_substs
            ),
        }
    }
}
