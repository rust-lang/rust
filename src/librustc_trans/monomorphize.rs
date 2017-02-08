// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::*;
use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::traits;
use rustc::ty::fold::{TypeFolder, TypeFoldable};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::util::common::MemoizationMap;

use syntax::codemap::DUMMY_SP;

pub use rustc::ty::Instance;

/// For associated constants from traits, return the impl definition.
pub fn resolve_const<'a, 'tcx>(
    scx: &SharedCrateContext<'a, 'tcx>, def_id: DefId, substs: &'tcx Substs<'tcx>
) -> Instance<'tcx> {
    if let Some(trait_id) = scx.tcx().trait_of_item(def_id) {
        let trait_ref = ty::TraitRef::new(trait_id, substs);
        let trait_ref = ty::Binder(trait_ref);
        let vtable = fulfill_obligation(scx, DUMMY_SP, trait_ref);
        if let traits::VtableImpl(vtable_impl) = vtable {
            let name = scx.tcx().item_name(def_id);
            let ac = scx.tcx().associated_items(vtable_impl.impl_def_id)
                .find(|item| item.kind == ty::AssociatedKind::Const && item.name == name);
            if let Some(ac) = ac {
                return Instance::new(ac.def_id, vtable_impl.substs);
            }
        }
    }

    Instance::new(def_id, substs)
}

/// Monomorphizes a type from the AST by first applying the in-scope
/// substitutions and then normalizing any associated types.
pub fn apply_param_substs<'a, 'tcx, T>(scx: &SharedCrateContext<'a, 'tcx>,
                                       param_substs: &Substs<'tcx>,
                                       value: &T)
                                       -> T
    where T: TransNormalize<'tcx>
{
    let tcx = scx.tcx();
    debug!("apply_param_substs(param_substs={:?}, value={:?})", param_substs, value);
    let substituted = value.subst(tcx, param_substs);
    let substituted = scx.tcx().erase_regions(&substituted);
    AssociatedTypeNormalizer::new(scx).fold(&substituted)
}


/// Returns the normalized type of a struct field
pub fn field_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_substs: &Substs<'tcx>,
                          f: &'tcx ty::FieldDef)
                          -> Ty<'tcx>
{
    tcx.normalize_associated_type(&f.ty(tcx, param_substs))
}

struct AssociatedTypeNormalizer<'a, 'b: 'a, 'gcx: 'b> {
    shared: &'a SharedCrateContext<'b, 'gcx>,
}

impl<'a, 'b, 'gcx> AssociatedTypeNormalizer<'a, 'b, 'gcx> {
    fn new(shared: &'a SharedCrateContext<'b, 'gcx>) -> Self {
        AssociatedTypeNormalizer {
            shared: shared,
        }
    }

    fn fold<T:TypeFoldable<'gcx>>(&mut self, value: &T) -> T {
        if !value.has_projection_types() {
            value.clone()
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a, 'b, 'gcx> TypeFolder<'gcx, 'gcx> for AssociatedTypeNormalizer<'a, 'b, 'gcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'gcx, 'gcx> {
        self.shared.tcx()
    }

    fn fold_ty(&mut self, ty: Ty<'gcx>) -> Ty<'gcx> {
        if !ty.has_projection_types() {
            ty
        } else {
            self.shared.project_cache().memoize(ty, || {
                debug!("AssociatedTypeNormalizer: ty={:?}", ty);
                self.shared.tcx().normalize_associated_type(&ty)
            })
        }
    }
}
