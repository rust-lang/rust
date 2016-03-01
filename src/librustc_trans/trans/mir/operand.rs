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
use rustc::middle::ty::{self, Ty};
use rustc::mir::repr as mir;
use trans::adt;
use trans::base;
use trans::common::{self, Block, BlockAndBuilder};
use trans::datum;
use trans::Disr;
use trans::glue;

use super::{MirContext, TempRef, drop};
use super::lvalue::LvalueRef;

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone)]
pub enum OperandValue {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    Ref(ValueRef),
    /// A single LLVM value.
    Immediate(ValueRef),
    /// A fat pointer. The first ValueRef is the data and the second
    /// is the extra.
    FatPtr(ValueRef, ValueRef)
}

/// An `OperandRef` is an "SSA" reference to a Rust value, along with
/// its type.
///
/// NOTE: unless you know a value's type exactly, you should not
/// generate LLVM opcodes acting on it and instead act via methods,
/// to avoid nasty edge cases. In particular, using `Builder.store`
/// directly is sure to cause problems -- use `MirContext.store_operand`
/// instead.
#[derive(Copy, Clone)]
pub struct OperandRef<'tcx> {
    // The value.
    pub val: OperandValue,

    // The type of value being returned.
    pub ty: Ty<'tcx>
}

impl<'tcx> OperandRef<'tcx> {
    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> ValueRef {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => unreachable!()
        }
    }

    pub fn repr<'bcx>(self, bcx: &BlockAndBuilder<'bcx, 'tcx>) -> String {
        match self.val {
            OperandValue::Ref(r) => {
                format!("OperandRef(Ref({}) @ {:?})",
                        bcx.val_to_string(r), self.ty)
            }
            OperandValue::Immediate(i) => {
                format!("OperandRef(Immediate({}) @ {:?})",
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

    pub fn from_rvalue_datum(datum: datum::Datum<'tcx, datum::Rvalue>) -> OperandRef {
        OperandRef {
            ty: datum.ty,
            val: match datum.kind.mode {
                datum::RvalueMode::ByRef => OperandValue::Ref(datum.val),
                datum::RvalueMode::ByValue => OperandValue::Immediate(datum.val),
            }
        }
    }
}

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_load(&mut self,
                      bcx: &BlockAndBuilder<'bcx, 'tcx>,
                      llval: ValueRef,
                      ty: Ty<'tcx>)
                      -> OperandRef<'tcx>
    {
        debug!("trans_load: {} @ {:?}", bcx.val_to_string(llval), ty);

        let val = match datum::appropriate_rvalue_mode(bcx.ccx(), ty) {
            datum::ByValue => {
                bcx.with_block(|bcx| {
                    OperandValue::Immediate(base::load_ty(bcx, llval, ty))
                })
            }
            datum::ByRef if common::type_is_fat_ptr(bcx.tcx(), ty) => {
                let (lldata, llextra) = bcx.with_block(|bcx| {
                    base::load_fat_ptr(bcx, llval, ty)
                });
                OperandValue::FatPtr(lldata, llextra)
            }
            datum::ByRef => OperandValue::Ref(llval)
        };

        OperandRef { val: val, ty: ty }
    }

    pub fn trans_operand(&mut self,
                         bcx: &BlockAndBuilder<'bcx, 'tcx>,
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
                self.trans_load(bcx, tr_lvalue.llval, ty)
            }

            mir::Operand::Constant(ref constant) => {
                self.trans_constant(bcx, constant)
            }
        }
    }

    pub fn store_operand(&mut self,
                         bcx: &BlockAndBuilder<'bcx, 'tcx>,
                         lldest: ValueRef,
                         operand: OperandRef<'tcx>)
    {
        debug!("store_operand: operand={}", operand.repr(bcx));
        bcx.with_block(|bcx| self.store_operand_direct(bcx, lldest, operand))
    }

    pub fn store_operand_direct(&mut self,
                                bcx: Block<'bcx, 'tcx>,
                                lldest: ValueRef,
                                operand: OperandRef<'tcx>)
    {
        // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
        // value is through `undef`, and store itself is useless.
        if common::type_is_zero_size(bcx.ccx(), operand.ty) {
            return;
        }
        match operand.val {
            OperandValue::Ref(r) => base::memcpy_ty(bcx, lldest, r, operand.ty),
            OperandValue::Immediate(s) => base::store_ty(bcx, s, lldest, operand.ty),
            OperandValue::FatPtr(data, extra) => {
                base::store_fat_ptr(bcx, data, extra, lldest, operand.ty);
            }
        }
    }

    pub fn trans_operand_untupled(&mut self,
                                  bcx: &BlockAndBuilder<'bcx, 'tcx>,
                                  operand: &mir::Operand<'tcx>)
                                  -> Vec<OperandRef<'tcx>>
    {
        // FIXME: consider having some optimization to avoid tupling/untupling
        // (and storing/loading in the case of immediates)

        // avoid trans_operand for pointless copying
        let lv = match *operand {
            mir::Operand::Consume(ref lvalue) => self.trans_lvalue(bcx, lvalue),
            mir::Operand::Constant(ref constant) => {
                // FIXME: consider being less pessimized
                if constant.ty.is_nil() {
                    return vec![];
                }

                let ty = bcx.monomorphize(&constant.ty);
                let lv = LvalueRef::alloca(bcx, ty, "__untuple_alloca");
                let constant = self.trans_constant(bcx, constant);
                self.store_operand(bcx, lv.llval, constant);
                lv
           }
        };

        let lv_ty = lv.ty.to_ty(bcx.tcx());
        let result_types = match lv_ty.sty {
            ty::TyTuple(ref tys) => tys,
            _ => bcx.tcx().sess.span_bug(
                self.mir.span,
                &format!("bad final argument to \"rust-call\" fn {:?}", lv_ty))
        };

        let base_repr = adt::represent_type(bcx.ccx(), lv_ty);
        let base = adt::MaybeSizedValue::sized(lv.llval);
        result_types.iter().enumerate().map(|(n, &ty)| {
            self.trans_load(bcx, bcx.with_block(|bcx| {
                adt::trans_field_ptr(bcx, &base_repr, base, Disr(0), n)
            }), ty)
        }).collect()
    }

    pub fn set_operand_dropped(&mut self,
                               bcx: &BlockAndBuilder<'bcx, 'tcx>,
                               operand: &mir::Operand<'tcx>) {
        match *operand {
            mir::Operand::Constant(_) => return,
            mir::Operand::Consume(ref lvalue) => {
                if let mir::Lvalue::Temp(idx) = *lvalue {
                    if let TempRef::Operand(..) = self.temps[idx as usize] {
                        // All lvalues which have an associated drop are promoted to an alloca
                        // beforehand. If this is an operand, it is safe to say this is never
                        // dropped and there’s no reason for us to zero this out at all.
                        return
                    }
                }
                let lvalue = self.trans_lvalue(bcx, lvalue);
                let ty = lvalue.ty.to_ty(bcx.tcx());
                if !glue::type_needs_drop(bcx.tcx(), ty) {
                    return
                } else {
                    drop::drop_fill(bcx, lvalue.llval, ty);
                }
            }
        }
    }
}
