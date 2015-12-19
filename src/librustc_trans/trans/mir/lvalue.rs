// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use rustc::middle::ty::{self, Ty, HasTypeFlags};
use rustc::mir::repr as mir;
use rustc::mir::tcx::LvalueTy;
use trans::adt;
use trans::base;
use trans::build;
use trans::common::{self, Block};
use trans::debuginfo::DebugLoc;
use trans::machine;
use trans::type_of;
use llvm;

use std::ptr;

use super::{MirContext, TempRef};

#[derive(Copy, Clone)]
pub struct LvalueRef<'tcx> {
    /// Pointer to the contents of the lvalue
    pub llval: ValueRef,

    /// This lvalue's extra data if it is unsized, or null
    pub llextra: ValueRef,

    /// Monomorphized type of this lvalue, including variant information
    pub ty: LvalueTy<'tcx>,
}

impl<'tcx> LvalueRef<'tcx> {
    pub fn new_sized(llval: ValueRef, lvalue_ty: LvalueTy<'tcx>) -> LvalueRef<'tcx> {
        LvalueRef { llval: llval, llextra: ptr::null_mut(), ty: lvalue_ty }
    }

    pub fn alloca<'bcx>(bcx: Block<'bcx, 'tcx>,
                        ty: Ty<'tcx>,
                        name: &str)
                        -> LvalueRef<'tcx>
    {
        assert!(!ty.has_erasable_regions());
        let lltemp = base::alloc_ty(bcx, ty, name);
        LvalueRef::new_sized(lltemp, LvalueTy::from_ty(ty))
    }
}

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn lvalue_len(&mut self,
                      bcx: Block<'bcx, 'tcx>,
                      lvalue: LvalueRef<'tcx>)
                      -> ValueRef {
        match lvalue.ty.to_ty(bcx.tcx()).sty {
            ty::TyArray(_, n) => common::C_uint(bcx.ccx(), n),
            ty::TySlice(_) | ty::TyStr => {
                assert!(lvalue.llextra != ptr::null_mut());
                lvalue.llextra
            }
            _ => bcx.sess().bug("unexpected type in get_base_and_len"),
        }
    }

    pub fn trans_lvalue(&mut self,
                        bcx: Block<'bcx, 'tcx>,
                        lvalue: &mir::Lvalue<'tcx>)
                        -> LvalueRef<'tcx> {
        debug!("trans_lvalue(lvalue={:?})", lvalue);

        let fcx = bcx.fcx;
        let ccx = fcx.ccx;
        let tcx = bcx.tcx();
        match *lvalue {
            mir::Lvalue::Var(index) => self.vars[index as usize],
            mir::Lvalue::Temp(index) => match self.temps[index as usize] {
                TempRef::Lvalue(lvalue) =>
                    lvalue,
                TempRef::Operand(..) =>
                    tcx.sess.bug(&format!("using operand temp {:?} as lvalue", lvalue)),
            },
            mir::Lvalue::Arg(index) => self.args[index as usize],
            mir::Lvalue::Static(def_id) => {
                let const_ty = self.mir.lvalue_ty(tcx, lvalue);
                LvalueRef::new_sized(
                    common::get_static_val(ccx, def_id, const_ty.to_ty(tcx)),
                    const_ty)
            },
            mir::Lvalue::ReturnPointer => {
                let fn_return_ty = bcx.monomorphize(&self.mir.return_ty);
                let return_ty = fn_return_ty.unwrap();
                let llval = if !common::return_type_is_void(bcx.ccx(), return_ty) {
                    fcx.get_ret_slot(bcx, fn_return_ty, "")
                } else {
                    // This is a void return; that is, there’s no place to store the value and
                    // there cannot really be one (or storing into it doesn’t make sense, anyway).
                    // Ergo, we return an undef ValueRef, so we do not have to special-case every
                    // place using lvalues, and could use it the same way you use a regular
                    // ReturnPointer LValue (i.e. store into it, load from it etc).
                    let llty = type_of::type_of(bcx.ccx(), return_ty).ptr_to();
                    unsafe {
                        llvm::LLVMGetUndef(llty.to_ref())
                    }
                };
                LvalueRef::new_sized(llval, LvalueTy::from_ty(return_ty))
            },
            mir::Lvalue::Projection(ref projection) => {
                let tr_base = self.trans_lvalue(bcx, &projection.base);
                let projected_ty = tr_base.ty.projection_ty(tcx, &projection.elem);
                let (llprojected, llextra) = match projection.elem {
                    mir::ProjectionElem::Deref => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        if common::type_is_sized(tcx, projected_ty.to_ty(tcx)) {
                            (base::load_ty(bcx, tr_base.llval, base_ty),
                             ptr::null_mut())
                        } else {
                            base::load_fat_ptr(bcx, tr_base.llval, base_ty)
                        }
                    }
                    mir::ProjectionElem::Field(ref field) => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        let base_repr = adt::represent_type(ccx, base_ty);
                        let discr = match tr_base.ty {
                            LvalueTy::Ty { .. } => 0,
                            LvalueTy::Downcast { adt_def: _, substs: _, variant_index: v } => v,
                        };
                        let discr = discr as u64;
                        let is_sized = common::type_is_sized(tcx, projected_ty.to_ty(tcx));
                        let base = if is_sized {
                            adt::MaybeSizedValue::sized(tr_base.llval)
                        } else {
                            adt::MaybeSizedValue::unsized_(tr_base.llval, tr_base.llextra)
                        };
                        (adt::trans_field_ptr(bcx, &base_repr, base, discr, field.index()),
                         if is_sized {
                             ptr::null_mut()
                         } else {
                             tr_base.llextra
                         })
                    }
                    mir::ProjectionElem::Index(ref index) => {
                        let index = self.trans_operand(bcx, index);
                        let llindex = self.prepare_index(bcx, index.immediate());
                        let zero = common::C_uint(bcx.ccx(), 0u64);
                        (build::InBoundsGEP(bcx, tr_base.llval, &[zero, llindex]),
                         ptr::null_mut())
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = common::C_u32(bcx.ccx(), offset);
                        let llindex = self.prepare_index(bcx, lloffset);
                        let zero = common::C_uint(bcx.ccx(), 0u64);
                        (build::InBoundsGEP(bcx, tr_base.llval, &[zero, llindex]),
                         ptr::null_mut())
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = common::C_u32(bcx.ccx(), offset);
                        let lllen = self.lvalue_len(bcx, tr_base);
                        let llindex = build::Sub(bcx, lllen, lloffset, DebugLoc::None);
                        let llindex = self.prepare_index(bcx, llindex);
                        let zero = common::C_uint(bcx.ccx(), 0u64);
                        (build::InBoundsGEP(bcx, tr_base.llval, &[zero, llindex]),
                         ptr::null_mut())
                    }
                    mir::ProjectionElem::Downcast(..) => {
                        (tr_base.llval, tr_base.llextra)
                    }
                };
                LvalueRef {
                    llval: llprojected,
                    llextra: llextra,
                    ty: projected_ty,
                }
            }
        }
    }

    /// Adjust the bitwidth of an index since LLVM is less forgiving
    /// than we are.
    ///
    /// nmatsakis: is this still necessary? Not sure.
    fn prepare_index(&mut self,
                     bcx: Block<'bcx, 'tcx>,
                     llindex: ValueRef)
                     -> ValueRef
    {
        let ccx = bcx.ccx();
        let index_size = machine::llbitsize_of_real(bcx.ccx(), common::val_ty(llindex));
        let int_size = machine::llbitsize_of_real(bcx.ccx(), ccx.int_type());
        if index_size < int_size {
            build::ZExt(bcx, llindex, ccx.int_type())
        } else if index_size > int_size {
            build::Trunc(bcx, llindex, ccx.int_type())
        } else {
            llindex
        }
    }
}
