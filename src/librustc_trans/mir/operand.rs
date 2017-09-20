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
use rustc::ty;
use rustc::ty::layout::{LayoutOf, FullLayout};
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;

use base;
use common::{self, CrateContext, C_undef};
use builder::Builder;
use value::Value;
use type_of::LayoutLlvmExt;

use std::fmt;
use std::ptr;

use super::{MirContext, LocalRef};
use super::constant::Const;
use super::lvalue::{Alignment, LvalueRef};

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone)]
pub enum OperandValue {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    Ref(ValueRef, Alignment),
    /// A single LLVM value.
    Immediate(ValueRef),
    /// A pair of immediate LLVM values. Used by fat pointers too.
    Pair(ValueRef, ValueRef)
}

impl fmt::Debug for OperandValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OperandValue::Ref(r, align) => {
                write!(f, "Ref({:?}, {:?})", Value(r), align)
            }
            OperandValue::Immediate(i) => {
                write!(f, "Immediate({:?})", Value(i))
            }
            OperandValue::Pair(a, b) => {
                write!(f, "Pair({:?}, {:?})", Value(a), Value(b))
            }
        }
    }
}

/// An `OperandRef` is an "SSA" reference to a Rust value, along with
/// its type.
///
/// NOTE: unless you know a value's type exactly, you should not
/// generate LLVM opcodes acting on it and instead act via methods,
/// to avoid nasty edge cases. In particular, using `Builder::store`
/// directly is sure to cause problems -- use `OperandRef::store`
/// instead.
#[derive(Copy, Clone)]
pub struct OperandRef<'tcx> {
    // The value.
    pub val: OperandValue,

    // The layout of value, based on its Rust type.
    pub layout: FullLayout<'tcx>,
}

impl<'tcx> fmt::Debug for OperandRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, 'tcx> OperandRef<'tcx> {
    pub fn new_zst(ccx: &CrateContext<'a, 'tcx>,
                   layout: FullLayout<'tcx>) -> OperandRef<'tcx> {
        assert!(layout.is_zst());
        let llty = layout.llvm_type(ccx);
        // FIXME(eddyb) ZSTs should always be immediate, not pairs.
        // This hack only exists to unpack a constant undef pair.
        Const::new(C_undef(llty), layout.ty).to_operand(ccx)
    }

    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> ValueRef {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self)
        }
    }

    pub fn deref(self, ccx: &CrateContext<'a, 'tcx>) -> LvalueRef<'tcx> {
        let projected_ty = self.layout.ty.builtin_deref(true, ty::NoPreference)
            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", self)).ty;
        let (llptr, llextra) = match self.val {
            OperandValue::Immediate(llptr) => (llptr, ptr::null_mut()),
            OperandValue::Pair(llptr, llextra) => (llptr, llextra),
            OperandValue::Ref(..) => bug!("Deref of by-Ref operand {:?}", self)
        };
        LvalueRef {
            llval: llptr,
            llextra,
            layout: ccx.layout_of(projected_ty),
            alignment: Alignment::AbiAligned,
        }
    }

    /// If this operand is a Pair, we return an
    /// Immediate aggregate with the two values.
    pub fn pack_if_pair(mut self, bcx: &Builder<'a, 'tcx>) -> OperandRef<'tcx> {
        if let OperandValue::Pair(a, b) = self.val {
            let llty = self.layout.llvm_type(bcx.ccx);
            debug!("Operand::pack_if_pair: packing {:?} into {:?}", self, llty);
            // Reconstruct the immediate aggregate.
            let mut llpair = C_undef(llty);
            let elems = [a, b];
            for i in 0..2 {
                let elem = base::from_immediate(bcx, elems[i]);
                llpair = bcx.insert_value(llpair, elem, self.layout.llvm_field_index(i));
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
            if common::type_is_imm_pair(bcx.ccx, self.layout.ty) {
                debug!("Operand::unpack_if_pair: unpacking {:?}", self);

                let a = bcx.extract_value(llval, self.layout.llvm_field_index(0));
                let a = base::to_immediate(bcx, a, self.layout.field(bcx.ccx, 0));

                let b = bcx.extract_value(llval, self.layout.llvm_field_index(1));
                let b = base::to_immediate(bcx, b, self.layout.field(bcx.ccx, 1));

                self.val = OperandValue::Pair(a, b);
            }
        }
        self
    }
}

impl<'a, 'tcx> OperandValue {
    pub fn store(self, bcx: &Builder<'a, 'tcx>, dest: LvalueRef<'tcx>) {
        debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);
        // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
        // value is through `undef`, and store itself is useless.
        if dest.layout.is_zst() {
            return;
        }
        match self {
            OperandValue::Ref(r, source_align) =>
                base::memcpy_ty(bcx, dest.llval, r, dest.layout,
                                (source_align | dest.alignment).non_abi()),
            OperandValue::Immediate(s) => {
                bcx.store(base::from_immediate(bcx, s), dest.llval, dest.alignment.non_abi());
            }
            OperandValue::Pair(a, b) => {
                // See comment above about zero-sized values.
                let dest_a = dest.project_field(bcx, 0);
                if !dest_a.layout.is_zst() {
                    let a = base::from_immediate(bcx, a);
                    bcx.store(a, dest_a.llval, dest_a.alignment.non_abi());
                }
                let dest_b = dest.project_field(bcx, 1);
                if !dest_b.layout.is_zst() {
                    let b = base::from_immediate(bcx, b);
                    bcx.store(b, dest_b.llval, dest_b.alignment.non_abi());
                }
            }
        }
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
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
                                layout: bcx.ccx.layout_of(self.monomorphize(&ty))
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
        self.trans_lvalue(bcx, lvalue).load(bcx)
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
                let val = self.trans_constant(&bcx, constant);
                let operand = val.to_operand(bcx.ccx);
                if let OperandValue::Ref(ptr, align) = operand.val {
                    // If this is a OperandValue::Ref to an immediate constant, load it.
                    LvalueRef::new_sized(ptr, operand.layout, align).load(bcx)
                } else {
                    operand
                }
            }
        }
    }
}
