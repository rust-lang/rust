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
use trans::common::{self, Block};
use trans::datum;

use super::{MirContext, TempRef};

/// The Rust representation of an operand's value. This is uniquely
/// determined by the operand type, but is kept as an enum as a
/// safety check.
#[derive(Copy, Clone)]
pub enum OperandValue {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    Ref(ValueRef),
    /// A single LLVM value.
    Imm(ValueRef),
    /// A fat pointer. The first ValueRef is the data and the second
    /// is the extra.
    FatPtr(ValueRef, ValueRef)
}

#[derive(Copy, Clone)]
pub struct OperandRef<'tcx> {
    // This will be "indirect" if `appropriate_rvalue_mode` returns
    // ByRef, and otherwise ByValue.
    pub val: OperandValue,

    // The type of value being returned.
    pub ty: Ty<'tcx>
}

impl<'tcx> OperandRef<'tcx> {
    pub fn immediate(self) -> ValueRef {
        match self.val {
            OperandValue::Imm(s) => s,
            _ => unreachable!()
        }
    }

    pub fn repr<'bcx>(self, bcx: Block<'bcx, 'tcx>) -> String {
        match self.val {
            OperandValue::Ref(r) => {
                format!("OperandRef(Ref({}) @ {:?})",
                        bcx.val_to_string(r), self.ty)
            }
            OperandValue::Imm(i) => {
                format!("OperandRef(Imm({}) @ {:?})",
                        bcx.val_to_string(i), self.ty)
            }
            OperandValue::FatPtr(a, d) => {
                format!("OperandRef(FatPtr({}, {}) @ {:?})",
                        bcx.val_to_string(a),
                        bcx.val_to_string(d),
                        self.ty)
            }
        }
    }
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
                let val = match datum::appropriate_rvalue_mode(bcx.ccx(), ty) {
                    datum::ByValue => {
                        OperandValue::Imm(base::load_ty(bcx, tr_lvalue.llval, ty))
                    }
                    datum::ByRef if common::type_is_fat_ptr(bcx.tcx(), ty) => {
                        let (lldata, llextra) = base::load_fat_ptr(bcx, tr_lvalue.llval, ty);
                        OperandValue::FatPtr(lldata, llextra)
                    }
                    datum::ByRef => OperandValue::Ref(tr_lvalue.llval)
                };
                OperandRef {
                    val: val,
                    ty: ty
                }
            }

            mir::Operand::Constant(ref constant) => {
                self.trans_constant(bcx, constant)
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

        // FIXME: consider not copying constants through the
        // stack.

        let o = self.trans_operand(bcx, operand);
        self.store_operand(bcx, lldest, o);
    }

    pub fn store_operand(&mut self,
                         bcx: Block<'bcx, 'tcx>,
                         lldest: ValueRef,
                         operand: OperandRef<'tcx>)
    {
        debug!("store_operand: operand={}", operand.repr(bcx));
        match operand.val {
            OperandValue::Ref(r) => base::memcpy_ty(bcx, lldest, r, operand.ty),
            OperandValue::Imm(s) => base::store_ty(bcx, s, lldest, operand.ty),
            OperandValue::FatPtr(data, extra) => {
                base::store_fat_ptr(bcx, data, extra, lldest, operand.ty);
            }
        }
    }
}
