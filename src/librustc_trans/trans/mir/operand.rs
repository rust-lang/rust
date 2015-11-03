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
use trans::base;
use trans::build;
use trans::common::Block;
use trans::datum;

use super::{MirContext, TempRef};

#[derive(Copy, Clone)]
pub struct OperandRef<'tcx> {
    // This will be "indirect" if `appropriate_rvalue_mode` returns
    // ByRef, and otherwise ByValue.
    pub llval: ValueRef,

    // The type of value being returned.
    pub ty: Ty<'tcx>
}

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_operand(&mut self,
                         bcx: Block<'bcx, 'tcx>,
                         operand: &mir::Operand<'tcx>)
                         -> OperandRef<'tcx>
    {
        debug!("trans_operand(operand={:?})", operand);

        match *operand {
            mir::Operand::Consume(ref lvalue) => {
                // watch out for temporaries that do not have an
                // alloca; they are handled somewhat differently
                if let &mir::Lvalue::Temp(index) = lvalue {
                    match self.temps[index as usize] {
                        TempRef::Operand(Some(o)) => {
                            return o;
                        }
                        TempRef::Operand(None) => {
                            bcx.tcx().sess.bug(
                                &format!("use of {:?} before def", lvalue));
                        }
                        TempRef::Lvalue(..) => {
                            // use path below
                        }
                    }
                }

                // for most lvalues, to consume them we just load them
                // out from their home
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                let ty = tr_lvalue.ty.to_ty(bcx.tcx());
                debug!("trans_operand: tr_lvalue={} @ {:?}",
                       bcx.val_to_string(tr_lvalue.llval),
                       ty);
                let llval = match datum::appropriate_rvalue_mode(bcx.ccx(), ty) {
                    datum::ByValue => build::Load(bcx, tr_lvalue.llval),
                    datum::ByRef => tr_lvalue.llval,
                };
                OperandRef {
                    llval: llval,
                    ty: ty
                }
            }

            mir::Operand::Constant(ref constant) => {
                let llval = self.trans_constant(bcx, constant);
                let ty = bcx.monomorphize(&constant.ty);
                OperandRef {
                    llval: llval,
                    ty: ty,
                }
            }
        }
    }

    pub fn trans_operand_into(&mut self,
                              bcx: Block<'bcx, 'tcx>,
                              lldest: ValueRef,
                              operand: &mir::Operand<'tcx>)
    {
        debug!("trans_operand_into(lldest={}, operand={:?})",
               bcx.val_to_string(lldest),
               operand);

        match *operand {
            mir::Operand::Consume(ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                let lvalue_ty = tr_lvalue.ty.to_ty(bcx.tcx());
                debug!("trans_operand_into: tr_lvalue={} @ {:?}",
                       bcx.val_to_string(tr_lvalue.llval),
                       lvalue_ty);
                base::memcpy_ty(bcx, lldest, tr_lvalue.llval, lvalue_ty);
            }

            mir::Operand::Constant(..) => {
                unimplemented!()
            }
        }
    }
}
