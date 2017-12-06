// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types, non_snake_case)]

//! Code that is useful in various trans modules.

use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;
use rustc::traits;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;

use syntax::attr;
use syntax_pos::DUMMY_SP;

pub fn type_is_sized<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.is_sized(tcx, ty::ParamEnv::empty(traits::Reveal::All), DUMMY_SP)
}

pub fn type_has_metadata<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    if type_is_sized(tcx, ty) {
        return false;
    }

    let tail = tcx.struct_tail(ty);
    match tail.sty {
        ty::TyForeign(..) => false,
        ty::TyStr | ty::TySlice(..) | ty::TyDynamic(..) => true,
        _ => bug!("unexpected unsized tail: {:?}", tail.sty),
    }
}

pub fn requests_inline<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: &ty::Instance<'tcx>
) -> bool {
    if is_inline_instance(tcx, instance) {
        return true
    }
    if let ty::InstanceDef::DropGlue(..) = instance.def {
        // Drop glue wants to be instantiated at every translation
        // unit, but without an #[inline] hint. We should make this
        // available to normal end-users.
        return true
    }
    attr::requests_inline(&instance.def.attrs(tcx)[..]) ||
        tcx.is_const_fn(instance.def.def_id())
}

pub fn is_inline_instance<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: &ty::Instance<'tcx>
) -> bool {
    let def_id = match instance.def {
        ty::InstanceDef::Item(def_id) => def_id,
        ty::InstanceDef::DropGlue(_, Some(_)) => return false,
        _ => return true
    };
    match tcx.def_key(def_id).disambiguated_data.data {
        DefPathData::StructCtor |
        DefPathData::EnumVariant(..) |
        DefPathData::ClosureExpr => true,
        _ => false
    }
}

/// Given a DefId and some Substs, produces the monomorphic item type.
pub fn def_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        def_id: DefId,
                        substs: &'tcx Substs<'tcx>)
                        -> Ty<'tcx>
{
    let ty = tcx.type_of(def_id);
    tcx.trans_apply_param_substs(substs, &ty)
}

/// Return the substituted type of an instance.
pub fn instance_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             instance: &ty::Instance<'tcx>)
                             -> Ty<'tcx>
{
    let ty = instance.def.def_ty(tcx);
    tcx.trans_apply_param_substs(instance.substs, &ty)
}
