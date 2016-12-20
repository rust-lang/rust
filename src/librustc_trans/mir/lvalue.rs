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
use rustc::mir;
use rustc::mir::tcx::LvalueTy;
use rustc_data_structures::indexed_vec::Idx;
use adt;
use base;
use common::{self, BlockAndBuilder, CrateContext, C_uint, C_undef};
use consts;
use machine;
use type_of::type_of;
use type_of;
use Disr;

use std::ptr;

use super::{MirContext, LocalRef};
use super::operand::OperandValue;

#[derive(Copy, Clone, Debug)]
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

    pub fn alloca<'a>(bcx: &BlockAndBuilder<'a, 'tcx>,
                        ty: Ty<'tcx>,
                        name: &str)
                        -> LvalueRef<'tcx>
    {
        assert!(!ty.has_erasable_regions());
        let lltemp = base::alloc_ty(bcx, ty, name);
        LvalueRef::new_sized(lltemp, LvalueTy::from_ty(ty))
    }

    pub fn len<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> ValueRef {
        let ty = self.ty.to_ty(ccx.tcx());
        match ty.sty {
            ty::TyArray(_, n) => common::C_uint(ccx, n),
            ty::TySlice(_) | ty::TyStr => {
                assert!(self.llextra != ptr::null_mut());
                self.llextra
            }
            _ => bug!("unexpected type `{}` in LvalueRef::len", ty)
        }
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_lvalue(&mut self,
                        bcx: &BlockAndBuilder<'a, 'tcx>,
                        lvalue: &mir::Lvalue<'tcx>)
                        -> LvalueRef<'tcx> {
        debug!("trans_lvalue(lvalue={:?})", lvalue);

        let ccx = bcx.ccx;
        let tcx = bcx.tcx();

        if let mir::Lvalue::Local(index) = *lvalue {
            match self.locals[index] {
                LocalRef::Lvalue(lvalue) => {
                    return lvalue;
                }
                LocalRef::Operand(..) => {
                    bug!("using operand local {:?} as lvalue", lvalue);
                }
            }
        }

        let result = match *lvalue {
            mir::Lvalue::Local(_) => bug!(), // handled above
            mir::Lvalue::Static(def_id) => {
                let const_ty = self.monomorphized_lvalue_ty(lvalue);
                LvalueRef::new_sized(consts::get_static(ccx, def_id),
                                     LvalueTy::from_ty(const_ty))
            },
            mir::Lvalue::Projection(box mir::Projection {
                ref base,
                elem: mir::ProjectionElem::Deref
            }) => {
                // Load the pointer from its location.
                let ptr = self.trans_consume(bcx, base);
                let projected_ty = LvalueTy::from_ty(ptr.ty)
                    .projection_ty(tcx, &mir::ProjectionElem::Deref);
                let projected_ty = self.monomorphize(&projected_ty);
                let (llptr, llextra) = match ptr.val {
                    OperandValue::Immediate(llptr) => (llptr, ptr::null_mut()),
                    OperandValue::Pair(llptr, llextra) => (llptr, llextra),
                    OperandValue::Ref(_) => bug!("Deref of by-Ref type {:?}", ptr.ty)
                };
                LvalueRef {
                    llval: llptr,
                    llextra: llextra,
                    ty: projected_ty,
                }
            }
            mir::Lvalue::Projection(ref projection) => {
                let tr_base = self.trans_lvalue(bcx, &projection.base);
                let projected_ty = tr_base.ty.projection_ty(tcx, &projection.elem);
                let projected_ty = self.monomorphize(&projected_ty);

                let project_index = |llindex| {
                    let element = if let ty::TySlice(_) = tr_base.ty.to_ty(tcx).sty {
                        // Slices already point to the array element type.
                        bcx.inbounds_gep(tr_base.llval, &[llindex])
                    } else {
                        let zero = common::C_uint(bcx.ccx, 0u64);
                        bcx.inbounds_gep(tr_base.llval, &[zero, llindex])
                    };
                    element
                };

                let (llprojected, llextra) = match projection.elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        let discr = match tr_base.ty {
                            LvalueTy::Ty { .. } => 0,
                            LvalueTy::Downcast { adt_def: _, substs: _, variant_index: v } => v,
                        };
                        let discr = discr as u64;
                        let is_sized = self.ccx.shared().type_is_sized(projected_ty.to_ty(tcx));
                        let base = if is_sized {
                            adt::MaybeSizedValue::sized(tr_base.llval)
                        } else {
                            adt::MaybeSizedValue::unsized_(tr_base.llval, tr_base.llextra)
                        };
                        let llprojected = adt::trans_field_ptr(bcx, base_ty, base, Disr(discr),
                            field.index());
                        let llextra = if is_sized {
                            ptr::null_mut()
                        } else {
                            tr_base.llextra
                        };
                        (llprojected, llextra)
                    }
                    mir::ProjectionElem::Index(ref index) => {
                        let index = self.trans_operand(bcx, index);
                        (project_index(self.prepare_index(bcx, index.immediate())), ptr::null_mut())
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = C_uint(bcx.ccx, offset);
                        (project_index(lloffset), ptr::null_mut())
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = C_uint(bcx.ccx, offset);
                        let lllen = tr_base.len(bcx.ccx);
                        let llindex = bcx.sub(lllen, lloffset);
                        (project_index(llindex), ptr::null_mut())
                    }
                    mir::ProjectionElem::Subslice { from, to } => {
                        let llindex = C_uint(bcx.ccx, from);
                        let llbase = project_index(llindex);

                        let base_ty = tr_base.ty.to_ty(bcx.tcx());
                        match base_ty.sty {
                            ty::TyArray(..) => {
                                // must cast the lvalue pointer type to the new
                                // array type (*[%_; new_len]).
                                let base_ty = self.monomorphized_lvalue_ty(lvalue);
                                let llbasety = type_of::type_of(bcx.ccx, base_ty).ptr_to();
                                let llbase = bcx.pointercast(llbase, llbasety);
                                (llbase, ptr::null_mut())
                            }
                            ty::TySlice(..) => {
                                assert!(tr_base.llextra != ptr::null_mut());
                                let lllen = bcx.sub(tr_base.llextra,
                                                    C_uint(bcx.ccx, from+to));
                                (llbase, lllen)
                            }
                            _ => bug!("unexpected type {:?} in Subslice", base_ty)
                        }
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
        };
        debug!("trans_lvalue(lvalue={:?}) => {:?}", lvalue, result);
        result
    }

    // Perform an action using the given Lvalue.
    // If the Lvalue is an empty LocalRef::Operand, then a temporary stack slot
    // is created first, then used as an operand to update the Lvalue.
    pub fn with_lvalue_ref<F, U>(&mut self, bcx: &BlockAndBuilder<'a, 'tcx>,
                                 lvalue: &mir::Lvalue<'tcx>, f: F) -> U
    where F: FnOnce(&mut Self, LvalueRef<'tcx>) -> U
    {
        if let mir::Lvalue::Local(index) = *lvalue {
            match self.locals[index] {
                LocalRef::Lvalue(lvalue) => f(self, lvalue),
                LocalRef::Operand(None) => {
                    let lvalue_ty = self.monomorphized_lvalue_ty(lvalue);
                    let lvalue = LvalueRef::alloca(bcx,
                                                   lvalue_ty,
                                                   "lvalue_temp");
                    let ret = f(self, lvalue);
                    let op = self.trans_load(bcx, lvalue.llval, lvalue_ty);
                    self.locals[index] = LocalRef::Operand(Some(op));
                    ret
                }
                LocalRef::Operand(Some(_)) => {
                    // See comments in LocalRef::new_operand as to why
                    // we always have Some in a ZST LocalRef::Operand.
                    let ty = self.monomorphized_lvalue_ty(lvalue);
                    if common::type_is_zero_size(bcx.ccx, ty) {
                        // Pass an undef pointer as no stores can actually occur.
                        let llptr = C_undef(type_of(bcx.ccx, ty).ptr_to());
                        f(self, LvalueRef::new_sized(llptr, LvalueTy::from_ty(ty)))
                    } else {
                        bug!("Lvalue local already set");
                    }
                }
            }
        } else {
            let lvalue = self.trans_lvalue(bcx, lvalue);
            f(self, lvalue)
        }
    }

    /// Adjust the bitwidth of an index since LLVM is less forgiving
    /// than we are.
    ///
    /// nmatsakis: is this still necessary? Not sure.
    fn prepare_index(&mut self,
                     bcx: &BlockAndBuilder<'a, 'tcx>,
                     llindex: ValueRef)
                     -> ValueRef
    {
        let ccx = bcx.ccx;
        let index_size = machine::llbitsize_of_real(bcx.ccx, common::val_ty(llindex));
        let int_size = machine::llbitsize_of_real(bcx.ccx, ccx.int_type());
        if index_size < int_size {
            bcx.zext(llindex, ccx.int_type())
        } else if index_size > int_size {
            bcx.trunc(llindex, ccx.int_type())
        } else {
            llindex
        }
    }

    pub fn monomorphized_lvalue_ty(&self, lvalue: &mir::Lvalue<'tcx>) -> Ty<'tcx> {
        let tcx = self.ccx.tcx();
        let lvalue_ty = lvalue.ty(&self.mir, tcx);
        self.monomorphize(&lvalue_ty.to_ty(tcx))
    }
}
