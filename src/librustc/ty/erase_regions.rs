// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::{self, Ty, TyCtxt};
use ty::fold::{TypeFolder, TypeFoldable};

pub(super) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        erase_regions_ty,
        ..*providers
    };
}

fn erase_regions_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    // NB: use `super_fold_with` here. If we used `fold_with`, it
    // could invoke the `erase_regions_ty` query recursively.
    ty.super_fold_with(&mut RegionEraserVisitor { tcx })
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Returns an equivalent value with all free regions removed (note
    /// that late-bound regions remain, because they are important for
    /// subtyping, but they are anonymized and normalized as well)..
    pub fn erase_regions<T>(self, value: &T) -> T
        where T : TypeFoldable<'tcx>
    {
        let value1 = value.fold_with(&mut RegionEraserVisitor { tcx: self });
        debug!("erase_regions({:?}) = {:?}", value, value1);
        value1
    }
}

struct RegionEraserVisitor<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionEraserVisitor<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let Some(ty_lifted) = self.tcx.lift_to_global(&ty) {
            self.tcx.erase_regions_ty(ty_lifted)
        } else {
            ty.super_fold_with(self)
        }
    }

    fn fold_binder<T>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T>
        where T : TypeFoldable<'tcx>
    {
        let u = self.tcx.anonymize_late_bound_regions(t);
        u.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        // because late-bound regions affect subtyping, we can't
        // erase the bound/free distinction, but we can replace
        // all free regions with 'erased.
        //
        // Note that we *CAN* replace early-bound regions -- the
        // type system never "sees" those, they get substituted
        // away. In codegen, they will always be erased to 'erased
        // whenever a substitution occurs.
        match *r {
            ty::ReLateBound(..) => r,
            _ => self.tcx.types.re_erased
        }
    }
}
