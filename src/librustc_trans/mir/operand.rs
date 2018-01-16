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
use rustc::middle::const_val::ConstEvalErr;
use rustc::mir;
use rustc::mir::interpret::Value as MiriValue;
use rustc::ty::layout::{self, Align, LayoutOf, TyLayout};
use rustc_data_structures::indexed_vec::Idx;

use base;
use common::{self, CodegenCx, C_null, C_undef, C_usize};
use builder::Builder;
use value::Value;
use type_of::LayoutLlvmExt;
use type_::Type;

use std::fmt;
use std::ptr;

use super::{FunctionCx, LocalRef};
use super::constant::{primval_to_llvm};
use super::place::PlaceRef;

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone)]
pub enum OperandValue {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    Ref(ValueRef, Align),
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
    pub fn new_zst(cx: &CodegenCx<'a, 'tcx>,
                   layout: TyLayout<'tcx>) -> OperandRef<'tcx> {
        assert!(layout.is_zst());
        OperandRef {
            val: OperandValue::Immediate(C_undef(layout.immediate_llvm_type(cx))),
            layout
        }
    }

    pub fn from_const(bx: &Builder<'a, 'tcx>,
                      miri_val: MiriValue,
                      ty: ty::Ty<'tcx>)
                      -> Result<OperandRef<'tcx>, ConstEvalErr<'tcx>> {
        let layout = bx.cx.layout_of(ty);

        if layout.is_zst() {
            return Ok(OperandRef::new_zst(bx.cx, layout));
        }

        let val = match miri_val {
            MiriValue::ByVal(x) => {
                let scalar = match layout.abi {
                    layout::Abi::Scalar(ref x) => x,
                    _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                };
                let llval = primval_to_llvm(
                    bx.cx,
                    x,
                    scalar,
                    layout.immediate_llvm_type(bx.cx),
                );
                OperandValue::Immediate(llval)
            },
            MiriValue::ByValPair(a, b) => {
                let (a_scalar, b_scalar) = match layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (a, b),
                    _ => bug!("from_const: invalid ByValPair layout: {:#?}", layout)
                };
                let a_llval = primval_to_llvm(
                    bx.cx,
                    a,
                    a_scalar,
                    layout.scalar_pair_element_llvm_type(bx.cx, 0),
                );
                let b_llval = primval_to_llvm(
                    bx.cx,
                    b,
                    b_scalar,
                    layout.scalar_pair_element_llvm_type(bx.cx, 1),
                );
                OperandValue::Pair(a_llval, b_llval)
            },
            MiriValue::ByRef(ptr, align) => {
                let scalar = layout::Scalar {
                    value: layout::Primitive::Pointer,
                    valid_range: 0..=!0
                };
                let ptr = primval_to_llvm(
                    bx.cx,
                    ptr.into_inner_primval(),
                    &scalar,
                    layout.llvm_type(bx.cx).ptr_to(),
                );
                return Ok(PlaceRef::new_sized(ptr, layout, align).load(bx));
            },
        };

        Ok(OperandRef {
            val,
            layout
        })
    }

    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> ValueRef {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self)
        }
    }

    pub fn deref(self, cx: &CodegenCx<'a, 'tcx>) -> PlaceRef<'tcx> {
        let projected_ty = self.layout.ty.builtin_deref(true)
            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", self)).ty;
        let (llptr, llextra) = match self.val {
            OperandValue::Immediate(llptr) => (llptr, ptr::null_mut()),
            OperandValue::Pair(llptr, llextra) => (llptr, llextra),
            OperandValue::Ref(..) => bug!("Deref of by-Ref operand {:?}", self)
        };
        let layout = cx.layout_of(projected_ty);
        PlaceRef {
            llval: llptr,
            llextra,
            layout,
            align: layout.align,
        }
    }

    /// If this operand is a `Pair`, we return an aggregate with the two values.
    /// For other cases, see `immediate`.
    pub fn immediate_or_packed_pair(self, bx: &Builder<'a, 'tcx>) -> ValueRef {
        if let OperandValue::Pair(a, b) = self.val {
            let llty = self.layout.llvm_type(bx.cx);
            debug!("Operand::immediate_or_packed_pair: packing {:?} into {:?}",
                   self, llty);
            // Reconstruct the immediate aggregate.
            let mut llpair = C_undef(llty);
            llpair = bx.insert_value(llpair, a, 0);
            llpair = bx.insert_value(llpair, b, 1);
            llpair
        } else {
            self.immediate()
        }
    }

    /// If the type is a pair, we return a `Pair`, otherwise, an `Immediate`.
    pub fn from_immediate_or_packed_pair(bx: &Builder<'a, 'tcx>,
                                         llval: ValueRef,
                                         layout: TyLayout<'tcx>)
                                         -> OperandRef<'tcx> {
        let val = if layout.is_llvm_scalar_pair() {
            debug!("Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}",
                    llval, layout);

            // Deconstruct the immediate aggregate.
            OperandValue::Pair(bx.extract_value(llval, 0),
                               bx.extract_value(llval, 1))
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout }
    }

    pub fn extract_field(&self, bx: &Builder<'a, 'tcx>, i: usize) -> OperandRef<'tcx> {
        let field = self.layout.field(bx.cx, i);
        let offset = self.layout.fields.offset(i);

        let mut val = match (self.val, &self.layout.abi) {
            // If we're uninhabited, or the field is ZST, it has no data.
            _ if self.layout.abi == layout::Abi::Uninhabited || field.is_zst() => {
                return OperandRef {
                    val: OperandValue::Immediate(C_undef(field.immediate_llvm_type(bx.cx))),
                    layout: field
                };
            }

            // Newtype of a scalar, scalar pair or vector.
            (OperandValue::Immediate(_), _) |
            (OperandValue::Pair(..), _) if field.size == self.layout.size => {
                assert_eq!(offset.bytes(), 0);
                self.val
            }

            // Extract a scalar component from a pair.
            (OperandValue::Pair(a_llval, b_llval), &layout::Abi::ScalarPair(ref a, ref b)) => {
                if offset.bytes() == 0 {
                    assert_eq!(field.size, a.value.size(bx.cx));
                    OperandValue::Immediate(a_llval)
                } else {
                    assert_eq!(offset, a.value.size(bx.cx)
                        .abi_align(b.value.align(bx.cx)));
                    assert_eq!(field.size, b.value.size(bx.cx));
                    OperandValue::Immediate(b_llval)
                }
            }

            // `#[repr(simd)]` types are also immediate.
            (OperandValue::Immediate(llval), &layout::Abi::Vector { .. }) => {
                OperandValue::Immediate(
                    bx.extract_element(llval, C_usize(bx.cx, i as u64)))
            }

            _ => bug!("OperandRef::extract_field({:?}): not applicable", self)
        };

        // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
        match val {
            OperandValue::Immediate(ref mut llval) => {
                *llval = bx.bitcast(*llval, field.immediate_llvm_type(bx.cx));
            }
            OperandValue::Pair(ref mut a, ref mut b) => {
                *a = bx.bitcast(*a, field.scalar_pair_element_llvm_type(bx.cx, 0));
                *b = bx.bitcast(*b, field.scalar_pair_element_llvm_type(bx.cx, 1));
            }
            OperandValue::Ref(..) => bug!()
        }

        OperandRef {
            val,
            layout: field
        }
    }
}

impl<'a, 'tcx> OperandValue {
    pub fn store(self, bx: &Builder<'a, 'tcx>, dest: PlaceRef<'tcx>) {
        debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);
        // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
        // value is through `undef`, and store itself is useless.
        if dest.layout.is_zst() {
            return;
        }
        match self {
            OperandValue::Ref(r, source_align) =>
                base::memcpy_ty(bx, dest.llval, r, dest.layout,
                                source_align.min(dest.align)),
            OperandValue::Immediate(s) => {
                bx.store(base::from_immediate(bx, s), dest.llval, dest.align);
            }
            OperandValue::Pair(a, b) => {
                for (i, &x) in [a, b].iter().enumerate() {
                    let mut llptr = bx.struct_gep(dest.llval, i as u64);
                    // Make sure to always store i1 as i8.
                    if common::val_ty(x) == Type::i1(bx.cx) {
                        llptr = bx.pointercast(llptr, Type::i8p(bx.cx));
                    }
                    bx.store(base::from_immediate(bx, x), llptr, dest.align);
                }
            }
        }
    }
}

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    fn maybe_trans_consume_direct(&mut self,
                                  bx: &Builder<'a, 'tcx>,
                                  place: &mir::Place<'tcx>)
                                   -> Option<OperandRef<'tcx>>
    {
        debug!("maybe_trans_consume_direct(place={:?})", place);

        // watch out for locals that do not have an
        // alloca; they are handled somewhat differently
        if let mir::Place::Local(index) = *place {
            match self.locals[index] {
                LocalRef::Operand(Some(o)) => {
                    return Some(o);
                }
                LocalRef::Operand(None) => {
                    bug!("use of {:?} before def", place);
                }
                LocalRef::Place(..) => {
                    // use path below
                }
            }
        }

        // Moves out of scalar and scalar pair fields are trivial.
        if let &mir::Place::Projection(ref proj) = place {
            if let Some(o) = self.maybe_trans_consume_direct(bx, &proj.base) {
                match proj.elem {
                    mir::ProjectionElem::Field(ref f, _) => {
                        return Some(o.extract_field(bx, f.index()));
                    }
                    mir::ProjectionElem::Index(_) |
                    mir::ProjectionElem::ConstantIndex { .. } => {
                        // ZSTs don't require any actual memory access.
                        // FIXME(eddyb) deduplicate this with the identical
                        // checks in `trans_consume` and `extract_field`.
                        let elem = o.layout.field(bx.cx, 0);
                        if elem.is_zst() {
                            return Some(OperandRef::new_zst(bx.cx, elem));
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }

    pub fn trans_consume(&mut self,
                         bx: &Builder<'a, 'tcx>,
                         place: &mir::Place<'tcx>)
                         -> OperandRef<'tcx>
    {
        debug!("trans_consume(place={:?})", place);

        let ty = self.monomorphized_place_ty(place);
        let layout = bx.cx.layout_of(ty);

        // ZSTs don't require any actual memory access.
        if layout.is_zst() {
            return OperandRef::new_zst(bx.cx, layout);
        }

        if let Some(o) = self.maybe_trans_consume_direct(bx, place) {
            return o;
        }

        // for most places, to consume them we just load them
        // out from their home
        self.trans_place(bx, place).load(bx)
    }

    pub fn trans_operand(&mut self,
                         bx: &Builder<'a, 'tcx>,
                         operand: &mir::Operand<'tcx>)
                         -> OperandRef<'tcx>
    {
        debug!("trans_operand(operand={:?})", operand);

        match *operand {
            mir::Operand::Copy(ref place) |
            mir::Operand::Move(ref place) => {
                self.trans_consume(bx, place)
            }

            mir::Operand::Constant(ref constant) => {
                let ty = self.monomorphize(&constant.ty);
                self.mir_constant_to_miri_value(bx, constant)
                    .and_then(|c| OperandRef::from_const(bx, c, ty))
                    .unwrap_or_else(|err| {
                        err.report(bx.tcx(), constant.span, "const operand");
                        // We've errored, so we don't have to produce working code.
                        let layout = bx.cx.layout_of(ty);
                        PlaceRef::new_sized(
                            C_null(layout.llvm_type(bx.cx).ptr_to()),
                            layout,
                            layout.align,
                        ).load(bx)
                    })
            }
        }
    }
}
