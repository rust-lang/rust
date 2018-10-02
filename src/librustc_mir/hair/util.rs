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

    fn user_annotated_ty_for_adt(
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
}
