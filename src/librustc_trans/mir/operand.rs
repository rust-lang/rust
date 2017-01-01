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
use rustc::ty::Ty;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;

use base;
use common;
use builder::Builder;
use value::Value;
use type_of;
use type_::Type;

use std::fmt;

use super::{MirContext, LocalRef};

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
    /// A pair of immediate LLVM values. Used by fat pointers too.
    Pair(ValueRef, ValueRef)
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

impl<'tcx> fmt::Debug for OperandRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.val {
            OperandValue::Ref(r) => {
                write!(f, "OperandRef(Ref({:?}) @ {:?})",
                       Value(r), self.ty)
            }
            OperandValue::Immediate(i) => {
                write!(f, "OperandRef(Immediate({:?}) @ {:?})",
                       Value(i), self.ty)
            }
            OperandValue::Pair(a, b) => {
                write!(f, "OperandRef(Pair({:?}, {:?}) @ {:?})",
                       Value(a), Value(b), self.ty)
            }
        }
    }
}

impl<'a, 'tcx> OperandRef<'tcx> {
    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> ValueRef {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self)
        }
    }

    /// If this operand is a Pair, we return an
    /// Immediate aggregate with the two values.
    pub fn pack_if_pair(mut self, bcx: &Builder<'a, 'tcx>) -> OperandRef<'tcx> {
        if let OperandValue::Pair(a, b) = self.val {
            // Reconstruct the immediate aggregate.
            let llty = type_of::type_of(bcx.ccx, self.ty);
            let mut llpair = common::C_undef(llty);
            let elems = [a, b];
            for i in 0..2 {
                let mut elem = elems[i];
                // Extend boolean i1's to i8.
                if common::val_ty(elem) == Type::i1(bcx.ccx) {
                    elem = bcx.zext(elem, Type::i8(bcx.ccx));
                }
                llpair = bcx.insert_value(llpair, elem, i);
            }
            self.val = OperandValue::Immediate(llpair);
        }
        self
    }

    /// If this operand is a pair in an Immediate,
    /// we return a Pair with the two halves.
    pub fn unpack_if_pair(mut self, bcx: &Builder<'a, 'tcx>) -> OperandRef<'tcx> {
        if let OperandValue::Immediate(llval) = self.val {
            // Deconstruct the immediate aggregate.
            if common::type_is_imm_pair(bcx.ccx, self.ty) {
                debug!("Operand::unpack_if_pair: unpacking {:?}", self);

                let mut a = bcx.extract_value(llval, 0);
                let mut b = bcx.extract_value(llval, 1);

                let pair_fields = common::type_pair_fields(bcx.ccx, self.ty);
                if let Some([a_ty, b_ty]) = pair_fields {
                    if a_ty.is_bool() {
                        a = bcx.trunc(a, Type::i1(bcx.ccx));
                    }
                    if b_ty.is_bool() {
                        b = bcx.trunc(b, Type::i1(bcx.ccx));
                    }
                }

                self.val = OperandValue::Pair(a, b);
            }
        }
        self
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_load(&mut self,
                      bcx: &Builder<'a, 'tcx>,
                      llval: ValueRef,
                      ty: Ty<'tcx>)
                      -> OperandRef<'tcx>
    {
        debug!("trans_load: {:?} @ {:?}", Value(llval), ty);

        let val = if common::type_is_fat_ptr(bcx.ccx, ty) {
            let (lldata, llextra) = base::load_fat_ptr(bcx, llval, ty);
            OperandValue::Pair(lldata, llextra)
        } else if common::type_is_imm_pair(bcx.ccx, ty) {
            let [a_ty, b_ty] = common::type_pair_fields(bcx.ccx, ty).unwrap();
            let a_ptr = bcx.struct_gep(llval, 0);
            let b_ptr = bcx.struct_gep(llval, 1);

            OperandValue::Pair(
                base::load_ty(bcx, a_ptr, a_ty),
                base::load_ty(bcx, b_ptr, b_ty)
            )
        } else if common::type_is_immediate(bcx.ccx, ty) {
            OperandValue::Immediate(base::load_ty(bcx, llval, ty))
        } else {
            OperandValue::Ref(llval)
        };

        OperandRef { val: val, ty: ty }
    }

    pub fn trans_consume(&mut self,
                         bcx: &Builder<'a, 'tcx>,
                         lvalue: &mir::Lvalue<'tcx>)
                         -> OperandRef<'tcx>
    {
        debug!("trans_consume(lvalue={:?})", lvalue);

        // watch out for locals that do not have an
        // alloca; they are handled somewhat differently
        if let mir::Lvalue::Local(index) = *lvalue {
            match self.locals[index] {
                LocalRef::Operand(Some(o)) => {
                    return o;
                }
                LocalRef::Operand(None) => {
                    bug!("use of {:?} before def", lvalue);
                }
                LocalRef::Lvalue(..) => {
                    // use path below
                }
            }
        }

        // Moves out of pair fields are trivial.
        if let &mir::Lvalue::Projection(ref proj) = lvalue {
            if let mir::Lvalue::Local(index) = proj.base {
                if let LocalRef::Operand(Some(o)) = self.locals[index] {
                    match (o.val, &proj.elem) {
                        (OperandValue::Pair(a, b),
                         &mir::ProjectionElem::Field(ref f, ty)) => {
                            let llval = [a, b][f.index()];
                            let op = OperandRef {
                                val: OperandValue::Immediate(llval),
                                ty: self.monomorphize(&ty)
                            };

                            // Handle nested pairs.
                            return op.unpack_if_pair(bcx);
                        }
                        _ => {}
                    }
                }
            }
        }

        // for most lvalues, to consume them we just load them
        // out from their home
        let tr_lvalue = self.trans_lvalue(bcx, lvalue);
        let ty = tr_lvalue.ty.to_ty(bcx.tcx());
        self.trans_load(bcx, tr_lvalue.llval, ty)
    }

    pub fn trans_operand(&mut self,
                         bcx: &Builder<'a, 'tcx>,
                         operand: &mir::Operand<'tcx>)
                         -> OperandRef<'tcx>
    {
        debug!("trans_operand(operand={:?})", operand);

        match *operand {
            mir::Operand::Consume(ref lvalue) => {
                self.trans_consume(bcx, lvalue)
            }

            mir::Operand::Constant(ref constant) => {
                let val = self.trans_constant(bcx, constant);
                let operand = val.to_operand(bcx.ccx);
                if let OperandValue::Ref(ptr) = operand.val {
                    // If this is a OperandValue::Ref to an immediate constant, load it.
                    self.trans_load(bcx, ptr, operand.ty)
                } else {
                    operand
                }
            }
        }
    }

    pub fn store_operand(&mut self,
                         bcx: &Builder<'a, 'tcx>,
                         lldest: ValueRef,
                         operand: OperandRef<'tcx>,
                         align: Option<u32>) {
        debug!("store_operand: operand={:?}, align={:?}", operand, align);
        // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
        // value is through `undef`, and store itself is useless.
        if common::type_is_zero_size(bcx.ccx, operand.ty) {
            return;
        }
        match operand.val {
            OperandValue::Ref(r) => base::memcpy_ty(bcx, lldest, r, operand.ty, align),
            OperandValue::Immediate(s) => {
                bcx.store(base::from_immediate(bcx, s), lldest, align);
            }
            OperandValue::Pair(a, b) => {
                let a = base::from_immediate(bcx, a);
                let b = base::from_immediate(bcx, b);
                bcx.store(a, bcx.struct_gep(lldest, 0), align);
                bcx.store(b, bcx.struct_gep(lldest, 1), align);
            }
        }
    }
}
