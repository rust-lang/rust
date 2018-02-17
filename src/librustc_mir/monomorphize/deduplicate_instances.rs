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
use rustc::ty::{self, TyCtxt, Ty, ParamTy, TypeFoldable, Instance};
use rustc::ty::fold::TypeFolder;
use rustc::ty::subst::Kind;
use rustc::middle::const_val::ConstVal;
use rustc::mir::{Mir, Rvalue, Location};
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
        if let Some(ty) = subst.as_type() {
            let ty = match used_substs.parameters[ParamIdx(i as u32)] {
                ParamUsage::Unused => {
                    if false /*param.name.as_str().starts_with("<")*/ {
                        ty.into()
                    } else {
                        #[allow(unused_mut)]
                        let mut mir = Vec::new();
                        ::util::write_mir_pretty(tcx, Some(instance.def_id()), &mut mir).unwrap();
                        let mut generics = Some(tcx.generics_of(instance.def_id()));
                        let mut pretty_generics = String::new();
                        loop {
                            if let Some(ref gen) = generics {
                                for ty in &gen.types {
                                    pretty_generics.push_str(&format!("{}:{} at {:?}, ", ty.index, ty.name, tcx.def_span(ty.def_id)));
                                }
                            } else {
                                break;
                            }
                            generics = generics.and_then(|gen|gen.parent).map(|def_id|tcx.generics_of(def_id));
                        }
                        tcx.sess.warn(&format!("Unused subst {} for {:?}<{}>\n with mir: {}", i, instance, pretty_generics, String::from_utf8_lossy(&mir)));
                        tcx.mk_ty(ty::TyNever)
                    }
                }
                ParamUsage::LayoutUsed | ParamUsage::Used => ty.into(),
            };
            Kind::from(ty)
        } else {
            (*subst).clone()
        }
    }).collect::<Vec<_>>());
    info!("replace_unused_substs_with_ty_error(_) -> {:?}", instance);
    instance
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ParamIdx(u32);

impl ::rustc_data_structures::indexed_vec::Idx for ParamIdx {
    fn new(idx: usize) -> Self {
        assert!(idx < ::std::u32::MAX as usize);
        ParamIdx(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum ParamUsage {
    Unused = 0,
    #[allow(dead_code)]
    LayoutUsed = 1,
    Used = 2,
}

impl_stable_hash_for! { enum self::ParamUsage { Unused, LayoutUsed, Used} }

#[derive(Debug, Default, Clone)]
pub struct ParamsUsage {
    parameters: IndexVec<ParamIdx, ParamUsage>,
}

impl_stable_hash_for! { struct ParamsUsage { parameters } }

impl ParamsUsage {
    fn new(len: usize) -> ParamsUsage {
        ParamsUsage {
            parameters: IndexVec::from_elem_n(ParamUsage::Unused, len),
        }
    }
}

struct SubstsVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a>(
    TyCtxt<'a, 'gcx, 'tcx>,
    &'tcx Mir<'tcx>,
    ParamsUsage,
);

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> Visitor<'tcx> for SubstsVisitor<'a, 'gcx, 'tcx> {
    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        for promoted in &mir.promoted {
            self.visit_mir(promoted);
        }
        self.super_mir(mir);
    }

    fn visit_ty(&mut self, ty: &Ty<'tcx>, _: TyContext) {
        self.fold_ty(ty);
    }

    fn visit_const(&mut self, constant: &&'tcx ty::Const<'tcx>, _location: Location) {
        if let ConstVal::Unevaluated(_def_id, substs) = constant.val {
            for subst in substs {
                if let Some(ty) = subst.as_type() {
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
            /*ty::TyAdt(_, substs) => {
                for subst in substs {
                    if let Some(ty) = subst.as_type() {
                        ty.fold_with(self);
                    }
                }
            }
            ty::TyArray(ty, _) |
            ty::TySlice(ty) |
            ty::TyRawPtr(TypeAndMut { ty, .. }) |
            ty::TyRef(_, TypeAndMut { ty, .. }) => {
                ty.fold_with(self);
            }
            ty::TyFnDef(_, substs) => {
                for subst in substs {
                    if let Some(ty) = subst.as_type() {
                        ty.fold_with(self);
                    }
                }
            }
            ty::TyFnPtr(poly_fn_sig) => {
                for ty in poly_fn_sig.skip_binder().inputs_and_outputs {
                    ty.fold_with(self);
                }
            }
            ty::TyClosure(_, closure_substs) => {
                for subst in closure_substs.substs {
                    if let Some(ty) = subst.as_type() {
                        ty.fold_with(self);
                    }
                }
            }
            ty::TyGenerator(_, closure_substs, generator_interior) => {
                for subst in closure_substs.substs {
                    if let Some(ty) = subst.as_type() {
                        ty.fold_with(self);
                    }
                }
                generator_interior.witness.fold_with(self);
            }
            ty::TyTuple(types, _) => {
                for ty in types {
                    ty.fold_with(self);
                }
            }
            ty::TyProjection(projection_ty) => {
                for subst in projection_ty.substs {
                    if let Some(ty) = subst.as_type() {
                        ty.fold_with(self);
                    }
                }
            }*/
            ty::TyParam(param) => {
                self.2.parameters[ParamIdx(param.idx)] = ParamUsage::Used;
            }
            _ => {}
        }
        ty.super_fold_with(self)
    }
}

fn used_substs_for_instance<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a ,'tcx, 'tcx>,
    instance: Instance<'tcx>,
) -> ParamsUsage {
    let mir = tcx.instance_mir(instance.def);
    let generics = tcx.generics_of(instance.def_id());
    let sig = ::rustc::ty::ty_fn_sig(tcx, instance.ty(tcx));
    let sig = tcx.erase_late_bound_regions_and_normalize(&sig);
    let mut substs_visitor = SubstsVisitor(tcx, mir, ParamsUsage::new(instance.substs.len()));
    //substs_visitor.visit_mir(mir);
    mir.fold_with(&mut substs_visitor);
    for ty in sig.inputs().iter() {
        ty.fold_with(&mut substs_visitor);
    }
    for ty_param_def in &generics.types {
        if ParamTy::for_def(ty_param_def).is_self() {
            // The self parameter is important for trait selection
            (substs_visitor.2).parameters[ParamIdx(ty_param_def.index)] = ParamUsage::Used;
        }
    }
    sig.output().fold_with(&mut substs_visitor);
    substs_visitor.2
}
