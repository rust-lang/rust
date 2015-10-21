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
use rustc::middle::ty::Ty;
use rustc_mir::repr as mir;
use rustc_mir::tcx::LvalueTy;
use trans::adt;
use trans::base;
use trans::build;
use trans::common::{self, Block};
use trans::debuginfo::DebugLoc;
use trans::machine;
use trans::tvec;

use super::MirContext;

#[derive(Copy, Clone)]
pub struct LvalueRef<'tcx> {
    /// Pointer to the contents of the lvalue
    pub llval: ValueRef,

    /// Monomorphized type of this lvalue, including variant information
    pub ty: LvalueTy<'tcx>,
}

impl<'tcx> LvalueRef<'tcx> {
    pub fn new(llval: ValueRef, lvalue_ty: LvalueTy<'tcx>) -> LvalueRef<'tcx> {
        LvalueRef { llval: llval, ty: lvalue_ty }
    }

    pub fn alloca<'bcx>(bcx: Block<'bcx, 'tcx>,
                        ty: Ty<'tcx>,
                        name: &str)
                        -> LvalueRef<'tcx>
    {
        let lltemp = base::alloc_ty(bcx, ty, name);
        LvalueRef::new(lltemp, LvalueTy::from_ty(ty))
    }
}

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
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
            mir::Lvalue::Temp(index) => self.temps[index as usize],
            mir::Lvalue::Arg(index) => self.args[index as usize],
            mir::Lvalue::Static(_def_id) => unimplemented!(),
            mir::Lvalue::ReturnPointer => {
                let return_ty = bcx.monomorphize(&self.mir.return_ty);
                let llval = fcx.get_ret_slot(bcx, return_ty, "return");
                LvalueRef::new(llval, LvalueTy::from_ty(return_ty.unwrap()))
            }
            mir::Lvalue::Projection(ref projection) => {
                let tr_base = self.trans_lvalue(bcx, &projection.base);
                let projected_ty = tr_base.ty.projection_ty(tcx, &projection.elem);
                let llprojected = match projection.elem {
                    mir::ProjectionElem::Deref => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        base::load_ty(bcx, tr_base.llval, base_ty)
                    }
                    mir::ProjectionElem::Field(ref field) => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        let base_repr = adt::represent_type(ccx, base_ty);
                        let discr = match tr_base.ty {
                            LvalueTy::Ty { .. } => 0,
                            LvalueTy::Downcast { adt_def: _, substs: _, variant_index: v } => v,
                        };
                        let discr = discr as u64;
                        adt::trans_field_ptr(bcx, &base_repr, tr_base.llval, discr, field.index())
                    }
                    mir::ProjectionElem::Index(ref index) => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        let index = self.trans_operand(bcx, index);
                        let llindex = self.prepare_index(bcx, index.llval);
                        let (llbase, _) = tvec::get_base_and_len(bcx, tr_base.llval, base_ty);
                        build::InBoundsGEP(bcx, llbase, &[llindex])
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let base_ty = tr_base.ty.to_ty(tcx);
                        let lloffset = common::C_u32(bcx.ccx(), offset);
                        let llindex = self.prepare_index(bcx, lloffset);
                        let (llbase, _) = tvec::get_base_and_len(bcx,
                                                                 tr_base.llval,
                                                                 base_ty);
                        build::InBoundsGEP(bcx, llbase, &[llindex])
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = common::C_u32(bcx.ccx(), offset);
                        let base_ty = tr_base.ty.to_ty(tcx);
                        let (llbase, lllen) = tvec::get_base_and_len(bcx,
                                                                     tr_base.llval,
                                                                     base_ty);
                        let llindex = build::Sub(bcx, lllen, lloffset, DebugLoc::None);
                        let llindex = self.prepare_index(bcx, llindex);
                        build::InBoundsGEP(bcx, llbase, &[llindex])
                    }
                    mir::ProjectionElem::Downcast(..) => {
                        tr_base.llval
                    }
                };
                LvalueRef {
                    llval: llprojected,
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
