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
use rustc::ty::layout::{self, LayoutOf, TyLayout};
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;

use base;
use common::{CrateContext, C_undef, C_usize};
use builder::Builder;
use value::Value;
use type_of::LayoutLlvmExt;

use std::fmt;
use std::ptr;

use super::{MirContext, LocalRef};
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
    pub layout: TyLayout<'tcx>,
}

impl<'tcx> fmt::Debug for OperandRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, 'tcx> OperandRef<'tcx> {
    pub fn new_zst(ccx: &CrateContext<'a, 'tcx>,
                   layout: TyLayout<'tcx>) -> OperandRef<'tcx> {
        assert!(layout.is_zst());
        OperandRef {
            val: OperandValue::Immediate(C_undef(layout.llvm_type(ccx))),
            layout
        }
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

    /// If this operand is a `Pair`, we return an aggregate with the two values.
    /// For other cases, see `immediate`.
    pub fn immediate_or_packed_pair(self, bcx: &Builder<'a, 'tcx>) -> ValueRef {
        if let OperandValue::Pair(a, b) = self.val {
            let llty = self.layout.llvm_type(bcx.ccx);
            debug!("Operand::immediate_or_packed_pair: packing {:?} into {:?}",
                   self, llty);
            // Reconstruct the immediate aggregate.
            let mut llpair = C_undef(llty);
            llpair = bcx.insert_value(llpair, a, 0);
            llpair = bcx.insert_value(llpair, b, 1);
            llpair
        } else {
            self.immediate()
        }
    }

    /// If the type is a pair, we return a `Pair`, otherwise, an `Immediate`.
    pub fn from_immediate_or_packed_pair(bcx: &Builder<'a, 'tcx>,
                                         llval: ValueRef,
                                         layout: TyLayout<'tcx>)
                                         -> OperandRef<'tcx> {
        let val = if layout.is_llvm_scalar_pair() {
            debug!("Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}",
                    llval, layout);

            // Deconstruct the immediate aggregate.
            OperandValue::Pair(bcx.extract_value(llval, 0),
                               bcx.extract_value(llval, 1))
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout }
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
                for (i, &x) in [a, b].iter().enumerate() {
                    let field = dest.project_field(bcx, i);
                    // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
                    let x = bcx.bitcast(x, field.layout.immediate_llvm_type(bcx.ccx));
                    bcx.store(base::from_immediate(bcx, x),
                              field.llval, field.alignment.non_abi());
                }
            }
        }
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    fn maybe_trans_consume_direct(&mut self,
                                  bcx: &Builder<'a, 'tcx>,
                                  lvalue: &mir::Lvalue<'tcx>)
                                   -> Option<OperandRef<'tcx>>
    {
        debug!("maybe_trans_consume_direct(lvalue={:?})", lvalue);

        // watch out for locals that do not have an
        // alloca; they are handled somewhat differently
        if let mir::Lvalue::Local(index) = *lvalue {
            match self.locals[index] {
                LocalRef::Operand(Some(o)) => {
                    return Some(o);
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
            if let mir::ProjectionElem::Field(ref f, _) = proj.elem {
                if let Some(o) = self.maybe_trans_consume_direct(bcx, &proj.base) {
                    let layout = o.layout.field(bcx.ccx, f.index());
                    let offset = o.layout.fields.offset(f.index());

                    // Handled in `trans_consume`.
                    assert!(!layout.is_zst());

                    // Offset has to match a scalar component.
                    let llval = match (o.val, &o.layout.abi) {
                        (OperandValue::Immediate(llval),
                         &layout::Abi::Scalar(ref scalar)) => {
                            assert_eq!(offset.bytes(), 0);
                            assert_eq!(layout.size, scalar.value.size(bcx.ccx));
                            llval
                        }
                        (OperandValue::Pair(a_llval, b_llval),
                         &layout::Abi::ScalarPair(ref a, ref b)) => {
                            if offset.bytes() == 0 {
                                assert_eq!(layout.size, a.value.size(bcx.ccx));
                                a_llval
                            } else {
                                assert_eq!(offset, a.value.size(bcx.ccx)
                                    .abi_align(b.value.align(bcx.ccx)));
                                assert_eq!(layout.size, b.value.size(bcx.ccx));
                                b_llval
                            }
                        }

                        // `#[repr(simd)]` types are also immediate.
                        (OperandValue::Immediate(llval),
                         &layout::Abi::Vector) => {
                            bcx.extract_element(llval, C_usize(bcx.ccx, f.index() as u64))
                        }

                        _ => return None
                    };

                    // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
                    let llval = bcx.bitcast(llval, layout.immediate_llvm_type(bcx.ccx));
                    return Some(OperandRef {
                        val: OperandValue::Immediate(llval),
                        layout
                    });
                }
            }
        }

        None
    }

    pub fn trans_consume(&mut self,
                         bcx: &Builder<'a, 'tcx>,
                         lvalue: &mir::Lvalue<'tcx>)
                         -> OperandRef<'tcx>
    {
        debug!("trans_consume(lvalue={:?})", lvalue);

        let ty = self.monomorphized_lvalue_ty(lvalue);
        let layout = bcx.ccx.layout_of(ty);

        // ZSTs don't require any actual memory access.
        if layout.is_zst() {
            return OperandRef::new_zst(bcx.ccx, layout);
        }

        if let Some(o) = self.maybe_trans_consume_direct(bcx, lvalue) {
            return o;
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
