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
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::mir::repr as mir;
use rustc::mir::tcx::LvalueTy;
use abi;
use adt;
use base;
use builder::Builder;
use common::{self, BlockAndBuilder, C_uint};
use consts;
use machine;
use mir::drop;
use llvm;
use Disr;

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

    pub fn alloca<'bcx>(bcx: &BlockAndBuilder<'bcx, 'tcx>,
                        ty: Ty<'tcx>,
                        name: &str)
                        -> LvalueRef<'tcx>
    {
        assert!(!ty.has_erasable_regions());
        let lltemp = bcx.with_block(|bcx| base::alloc_ty(bcx, ty, name));
        if bcx.fcx().type_needs_drop(ty) {
            drop::drop_fill(bcx, lltemp, ty);
        }
        LvalueRef::new_sized(lltemp, LvalueTy::from_ty(ty))
    }
}

pub fn get_meta(b: &Builder, fat_ptr: ValueRef) -> ValueRef {
    b.struct_gep(fat_ptr, abi::FAT_PTR_EXTRA)
}

pub fn get_dataptr(b: &Builder, fat_ptr: ValueRef) -> ValueRef {
    b.struct_gep(fat_ptr, abi::FAT_PTR_ADDR)
}

pub fn load_fat_ptr(b: &Builder, fat_ptr: ValueRef) -> (ValueRef, ValueRef) {
    (b.load(get_dataptr(b, fat_ptr)), b.load(get_meta(b, fat_ptr)))
}

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn lvalue_len(&mut self,
                      bcx: &BlockAndBuilder<'bcx, 'tcx>,
                      lvalue: LvalueRef<'tcx>)
                      -> ValueRef {
        match lvalue.ty.to_ty(bcx.tcx()).sty {
            ty::TyArray(_, n) => common::C_uint(bcx.ccx(), n),
            ty::TySlice(_) | ty::TyStr => {
                assert!(lvalue.llextra != ptr::null_mut());
                lvalue.llextra
            }
            _ => bug!("unexpected type in lvalue_len"),
        }
    }

    pub fn trans_lvalue(&mut self,
                        bcx: &BlockAndBuilder<'bcx, 'tcx>,
                        lvalue: &mir::Lvalue<'tcx>)
                        -> LvalueRef<'tcx> {
        debug!("trans_lvalue(lvalue={:?})", lvalue);

        let fcx = bcx.fcx();
        let ccx = bcx.ccx();
        let tcx = bcx.tcx();
        match *lvalue {
            mir::Lvalue::Var(index) => self.vars[index as usize],
            mir::Lvalue::Temp(index) => match self.temps[index as usize] {
                TempRef::Lvalue(lvalue) =>
                    lvalue,
                TempRef::Operand(..) =>
                    bug!("using operand temp {:?} as lvalue", lvalue),
            },
            mir::Lvalue::Arg(index) => self.args[index as usize],
            mir::Lvalue::Static(def_id) => {
                let const_ty = self.mir.lvalue_ty(tcx, lvalue);
                LvalueRef::new_sized(consts::get_static(ccx, def_id).val, const_ty)
            },
            mir::Lvalue::ReturnPointer => {
                let llval = if !fcx.fn_ty.ret.is_ignore() {
                    bcx.with_block(|bcx| {
                        fcx.get_ret_slot(bcx, "")
                    })
                } else {
                    // This is a void return; that is, there’s no place to store the value and
                    // there cannot really be one (or storing into it doesn’t make sense, anyway).
                    // Ergo, we return an undef ValueRef, so we do not have to special-case every
                    // place using lvalues, and could use it the same way you use a regular
                    // ReturnPointer LValue (i.e. store into it, load from it etc).
                    let llty = fcx.fn_ty.ret.original_ty.ptr_to();
                    unsafe {
                        llvm::LLVMGetUndef(llty.to_ref())
                    }
                };
                let fn_return_ty = bcx.monomorphize(&self.mir.return_ty);
                let return_ty = fn_return_ty.unwrap();
                LvalueRef::new_sized(llval, LvalueTy::from_ty(return_ty))
            },
            mir::Lvalue::Projection(ref projection) => {
                let tr_base = self.trans_lvalue(bcx, &projection.base);
                let projected_ty = tr_base.ty.projection_ty(tcx, &projection.elem);
                let projected_ty = bcx.monomorphize(&projected_ty);

                let project_index = |llindex| {
                    let element = if let ty::TySlice(_) = tr_base.ty.to_ty(tcx).sty {
                        // Slices already point to the array element type.
                        bcx.inbounds_gep(tr_base.llval, &[llindex])
                    } else {
                        let zero = common::C_uint(bcx.ccx(), 0u64);
                        bcx.inbounds_gep(tr_base.llval, &[zero, llindex])
                    };
                    (element, ptr::null_mut())
                };

                let (llprojected, llextra) = match projection.elem {
                    mir::ProjectionElem::Deref => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        if common::type_is_sized(tcx, projected_ty.to_ty(tcx)) {
                            (base::load_ty_builder(bcx, tr_base.llval, base_ty),
                             ptr::null_mut())
                        } else {
                            load_fat_ptr(bcx, tr_base.llval)
                        }
                    }
                    mir::ProjectionElem::Field(ref field, _) => {
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
                        let llprojected = adt::trans_field_ptr_builder(bcx, &base_repr, base,
                                                                       Disr(discr), field.index());
                        let llextra = if is_sized {
                            ptr::null_mut()
                        } else {
                            tr_base.llextra
                        };
                        (llprojected, llextra)
                    }
                    mir::ProjectionElem::Index(ref index) => {
                        let index = self.trans_operand(bcx, index);
                        project_index(self.prepare_index(bcx, index.immediate()))
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = C_uint(bcx.ccx(), offset);
                        project_index(self.prepare_index(bcx, lloffset))
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = C_uint(bcx.ccx(), offset);
                        let lllen = self.lvalue_len(bcx, tr_base);
                        let llindex = bcx.sub(lllen, lloffset);
                        project_index(self.prepare_index(bcx, llindex))
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

    // Perform an action using the given Lvalue.
    // If the Lvalue is an empty TempRef::Operand, then a temporary stack slot
    // is created first, then used as an operand to update the Lvalue.
    pub fn with_lvalue_ref<F, U>(&mut self, bcx: &BlockAndBuilder<'bcx, 'tcx>,
                                 lvalue: &mir::Lvalue<'tcx>, f: F) -> U
    where F: FnOnce(&mut Self, LvalueRef<'tcx>) -> U
    {
        match *lvalue {
            mir::Lvalue::Temp(idx) => {
                match self.temps[idx as usize] {
                    TempRef::Lvalue(lvalue) => f(self, lvalue),
                    TempRef::Operand(None) => {
                        let lvalue_ty = self.mir.lvalue_ty(bcx.tcx(), lvalue);
                        let lvalue_ty = bcx.monomorphize(&lvalue_ty);
                        let lvalue = LvalueRef::alloca(bcx,
                                                       lvalue_ty.to_ty(bcx.tcx()),
                                                       "lvalue_temp");
                        let ret = f(self, lvalue);
                        let op = self.trans_load(bcx, lvalue.llval, lvalue_ty.to_ty(bcx.tcx()));
                        self.temps[idx as usize] = TempRef::Operand(Some(op));
                        ret
                    }
                    TempRef::Operand(Some(_)) => {
                        bug!("Lvalue temp already set");
                    }
                }
            }
            _ => {
                let lvalue = self.trans_lvalue(bcx, lvalue);
                f(self, lvalue)
            }
        }
    }

    /// Adjust the bitwidth of an index since LLVM is less forgiving
    /// than we are.
    ///
    /// nmatsakis: is this still necessary? Not sure.
    fn prepare_index(&mut self,
                     bcx: &BlockAndBuilder<'bcx, 'tcx>,
                     llindex: ValueRef)
                     -> ValueRef
    {
        let ccx = bcx.ccx();
        let index_size = machine::llbitsize_of_real(bcx.ccx(), common::val_ty(llindex));
        let int_size = machine::llbitsize_of_real(bcx.ccx(), ccx.int_type());
        if index_size < int_size {
            bcx.zext(llindex, ccx.int_type())
        } else if index_size > int_size {
            bcx.trunc(llindex, ccx.int_type())
        } else {
            llindex
        }
    }
}
