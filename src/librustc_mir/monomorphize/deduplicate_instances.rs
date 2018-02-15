// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::IndexVec;
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable, Instance, ParamTy};
use rustc::ty::fold::TypeFolder;
use rustc::ty::subst::{Kind, UnpackedKind};
use rustc::middle::const_val::ConstVal;
use rustc::mir::{Mir, Rvalue, Promoted, Location};
use rustc::mir::visit::{Visitor, TyContext};

/// Replace substs which aren't used by the function with TyError,
/// so that it doesn't end up in the binary multiple times
/// For example in the code
///
/// ```rust
/// fn foo<T>() { } // here, T is clearly unused =)
///
/// fn main() {
///     foo::<u32>();
///     foo::<u64>();
/// }
/// ```
///
/// `foo::<u32>` and `foo::<u64>` are collapsed to `foo::<{some dummy}>`,
/// because codegen for `foo` doesn't depend on the Subst for T.
pub(crate) fn collapse_interchangable_instances<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mut instance: Instance<'tcx>
) -> Instance<'tcx> {
    info!("replace_unused_substs_with_ty_error({:?})", instance);

    if instance.substs.is_noop() || !tcx.is_mir_available(instance.def_id()) {
        return instance;
    }
    match instance.ty(tcx).sty {
        ty::TyFnDef(def_id, _) => {
            //let attrs = tcx.item_attrs(def_id);
            if tcx.lang_items().items().iter().find(|l|**l == Some(def_id)).is_some() {
                return instance; // Lang items dont work otherwise
            }
        }
        _ => return instance, // Closures dont work otherwise
    }

    let used_substs = used_substs_for_instance(tcx, instance);
    instance.substs = tcx._intern_substs(&instance.substs.into_iter().enumerate().map(|(i, subst)| {
        if let UnpackedKind::Type(ty) = subst.unpack() {
            let ty = if used_substs.parameters.iter().find(|p|p.idx == i as u32).is_some() {
                ty.into()
            } else if let ty::TyParam(ref _param) = ty.sty {
                //^ Dont replace <closure_kind> and other internal params
                if false /*param.name.as_str().starts_with("<")*/ {
                    ty.into()
                } else {
                    tcx.sess.warn(&format!("Unused subst for {:?}", instance));
                    tcx.mk_ty(ty::TyNever)
                }
            } else {
                // Can't use TyError as it gives some ICE in rustc_trans::callee::get_fn
                tcx.sess.warn(&format!("Unused subst for {:?}", instance));
                tcx.mk_ty(ty::TyNever)
            };
            Kind::from(ty)
        } else {
            (*subst).clone()
        }
    }).collect::<Vec<_>>());
    info!("replace_unused_substs_with_ty_error(_) -> {:?}", instance);
    instance
}

#[derive(Debug, Default, Clone)]
pub struct UsedParameters {
    pub parameters: Vec<ParamTy>,
    pub promoted: IndexVec<Promoted, UsedParameters>,
}

impl_stable_hash_for! { struct UsedParameters { parameters, promoted } }

struct SubstsVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a>(
    TyCtxt<'a, 'gcx, 'tcx>,
    &'tcx Mir<'tcx>,
    UsedParameters,
);

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> Visitor<'tcx> for SubstsVisitor<'a, 'gcx, 'tcx> {
    fn visit_ty(&mut self, ty: &Ty<'tcx>, _: TyContext) {
        self.fold_ty(ty);
    }

    fn visit_const(&mut self, constant: &&'tcx ty::Const<'tcx>, _location: Location) {
        if let ConstVal::Unevaluated(_def_id, substs) = constant.val {
            for subst in substs {
                if let UnpackedKind::Type(ty) = subst.unpack() {
                    ty.fold_with(self);
                }
            }
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        let tcx = self.0;
        match *rvalue {
            Rvalue::Cast(_kind, ref op, ty) => {
                self.fold_ty(op.ty(&self.1.local_decls, tcx));
                self.fold_ty(ty);
            }
            _ => {}
        }
        self.super_rvalue(rvalue, location);
    }
}

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> TypeFolder<'gcx, 'tcx> for SubstsVisitor<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.0
    }
    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.needs_subst() {
            return ty;
        }
        match ty.sty {
            ty::TyParam(param) => {
                self.2.parameters.push(param);
                ty
            }
            ty::TyFnDef(_, substs) => {
                for subst in substs {
                    if let UnpackedKind::Type(ty) = subst.unpack() {
                        ty.fold_with(self);
                    }
                }
                ty.super_fold_with(self)
            }
            ty::TyClosure(_, closure_substs) => {
                for subst in closure_substs.substs {
                    if let UnpackedKind::Type(ty) = subst.unpack() {
                        ty.fold_with(self);
                    }
                }
                ty.super_fold_with(self)
            }
            _ => ty.super_fold_with(self)
        }
    }
}

fn used_substs_for_instance<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a ,'tcx, 'tcx>,
    instance: Instance<'tcx>,
) -> UsedParameters {
    let mir = tcx.instance_mir(instance.def);
    let sig = ::rustc::ty::ty_fn_sig(tcx, instance.ty(tcx));
    let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
    let mut substs_visitor = SubstsVisitor(tcx, mir, UsedParameters::default());
    substs_visitor.visit_mir(mir);
    for ty in sig.inputs().iter() {
        ty.fold_with(&mut substs_visitor);
    }
    sig.output().fold_with(&mut substs_visitor);
    let mut used_substs = substs_visitor.2;
    used_substs.parameters.sort_by_key(|s|s.idx);
    used_substs.parameters.dedup_by_key(|s|s.idx);
    used_substs.promoted = mir.promoted.iter().map(|mir| used_substs_for_mir(tcx, mir)).collect();
    used_substs
}

fn used_substs_for_mir<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a ,'tcx, 'tcx>,
    mir: &'tcx Mir<'tcx>,
) -> UsedParameters {
    let mut substs_visitor = SubstsVisitor(tcx, mir, UsedParameters::default());
    substs_visitor.visit_mir(mir);
    let mut used_substs = substs_visitor.2;
    used_substs.parameters.sort_by_key(|s|s.idx);
    used_substs.parameters.dedup_by_key(|s|s.idx);
    used_substs.promoted = mir.promoted.iter().map(|mir| used_substs_for_mir(tcx, mir)).collect();
    used_substs
}
