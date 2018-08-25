// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use std::convert::TryInto;

use rustc::mir;
use rustc::ty::layout::{self, Align, LayoutOf, TyLayout, HasDataLayout, IntegerExt};
use rustc_data_structures::indexed_vec::Idx;

use rustc::mir::interpret::{
    GlobalId, ConstValue, Scalar, EvalResult, Pointer, ScalarMaybeUndef, EvalErrorKind
};
use super::{EvalContext, Machine, MemPlace, MPlaceTy, PlaceExtra, MemoryKind};

/// A `Value` represents a single immediate self-contained Rust value.
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's codegen.
/// In particular, thanks to `ScalarPair`, arithmetic operations and casts can be entirely
/// defined on `Value`, and do not have to work with a `Place`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Value {
    Scalar(ScalarMaybeUndef),
    ScalarPair(ScalarMaybeUndef, ScalarMaybeUndef),
}

impl<'tcx> Value {
    pub fn new_slice(
        val: Scalar,
        len: u64,
        cx: impl HasDataLayout
    ) -> Self {
        Value::ScalarPair(val.into(), Scalar::Bits {
            bits: len as u128,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }.into())
    }

    pub fn new_dyn_trait(val: Scalar, vtable: Pointer) -> Self {
        Value::ScalarPair(val.into(), Scalar::Ptr(vtable).into())
    }

    #[inline]
    pub fn to_scalar_or_undef(self) -> ScalarMaybeUndef {
        match self {
            Value::Scalar(val) => val,
            Value::ScalarPair(..) => bug!("Got a fat pointer where a scalar was expected"),
        }
    }

    #[inline]
    pub fn to_scalar(self) -> EvalResult<'tcx, Scalar> {
        self.to_scalar_or_undef().not_undef()
    }

    /// Convert the value into a pointer (or a pointer-sized integer).
    /// Throws away the second half of a ScalarPair!
    #[inline]
    pub fn to_scalar_ptr(self) -> EvalResult<'tcx, Scalar> {
        match self {
            Value::Scalar(ptr) |
            Value::ScalarPair(ptr, _) => ptr.not_undef(),
        }
    }

    pub fn to_scalar_dyn_trait(self) -> EvalResult<'tcx, (Scalar, Pointer)> {
        match self {
            Value::ScalarPair(ptr, vtable) =>
                Ok((ptr.not_undef()?, vtable.to_ptr()?)),
            _ => bug!("expected ptr and vtable, got {:?}", self),
        }
    }

    pub fn to_scalar_slice(self, cx: impl HasDataLayout) -> EvalResult<'tcx, (Scalar, u64)> {
        match self {
            Value::ScalarPair(ptr, val) => {
                let len = val.to_bits(cx.data_layout().pointer_size)?;
                Ok((ptr.not_undef()?, len as u64))
            }
            _ => bug!("expected ptr and length, got {:?}", self),
        }
    }
}

// ScalarPair needs a type to interpret, so we often have a value and a type together
// as input for binary and cast operations.
#[derive(Copy, Clone, Debug)]
pub struct ValTy<'tcx> {
    pub value: Value,
    pub layout: TyLayout<'tcx>,
}

impl<'tcx> ::std::ops::Deref for ValTy<'tcx> {
    type Target = Value;
    #[inline(always)]
    fn deref(&self) -> &Value {
        &self.value
    }
}

/// An `Operand` is the result of computing a `mir::Operand`. It can be immediate,
/// or still in memory.  The latter is an optimization, to delay reading that chunk of
/// memory and to avoid having to store arbitrary-sized data here.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Operand {
    Immediate(Value),
    Indirect(MemPlace),
}

impl Operand {
    #[inline]
    pub fn from_ptr(ptr: Pointer, align: Align) -> Self {
        Operand::Indirect(MemPlace::from_ptr(ptr, align))
    }

    #[inline]
    pub fn from_scalar_value(val: Scalar) -> Self {
        Operand::Immediate(Value::Scalar(val.into()))
    }

    #[inline]
    pub fn to_mem_place(self) -> MemPlace {
        match self {
            Operand::Indirect(mplace) => mplace,
            _ => bug!("to_mem_place: expected Operand::Indirect, got {:?}", self),

        }
    }

    #[inline]
    pub fn to_immediate(self) -> Value {
        match self {
            Operand::Immediate(val) => val,
            _ => bug!("to_immediate: expected Operand::Immediate, got {:?}", self),

        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OpTy<'tcx> {
    pub op: Operand,
    pub layout: TyLayout<'tcx>,
}

impl<'tcx> ::std::ops::Deref for OpTy<'tcx> {
    type Target = Operand;
    #[inline(always)]
    fn deref(&self) -> &Operand {
        &self.op
    }
}

impl<'tcx> From<MPlaceTy<'tcx>> for OpTy<'tcx> {
    #[inline(always)]
    fn from(mplace: MPlaceTy<'tcx>) -> Self {
        OpTy {
            op: Operand::Indirect(*mplace),
            layout: mplace.layout
        }
    }
}

impl<'tcx> From<ValTy<'tcx>> for OpTy<'tcx> {
    #[inline(always)]
    fn from(val: ValTy<'tcx>) -> Self {
        OpTy {
            op: Operand::Immediate(val.value),
            layout: val.layout
        }
    }
}

impl<'tcx> OpTy<'tcx> {
    #[inline]
    pub fn from_ptr(ptr: Pointer, align: Align, layout: TyLayout<'tcx>) -> Self {
        OpTy { op: Operand::from_ptr(ptr, align), layout }
    }

    #[inline]
    pub fn from_aligned_ptr(ptr: Pointer, layout: TyLayout<'tcx>) -> Self {
        OpTy { op: Operand::from_ptr(ptr, layout.align), layout }
    }

    #[inline]
    pub fn from_scalar_value(val: Scalar, layout: TyLayout<'tcx>) -> Self {
        OpTy { op: Operand::Immediate(Value::Scalar(val.into())), layout }
    }
}

// Use the existing layout if given (but sanity check in debug mode),
// or compute the layout.
#[inline(always)]
fn from_known_layout<'tcx>(
    layout: Option<TyLayout<'tcx>>,
    compute: impl FnOnce() -> EvalResult<'tcx, TyLayout<'tcx>>
) -> EvalResult<'tcx, TyLayout<'tcx>> {
    match layout {
        None => compute(),
        Some(layout) => {
            if cfg!(debug_assertions) {
                let layout2 = compute()?;
                assert_eq!(layout.details, layout2.details,
                    "Mismatch in layout of supposedly equal-layout types {:?} and {:?}",
                    layout.ty, layout2.ty);
            }
            Ok(layout)
        }
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Try reading a value in memory; this is interesting particularily for ScalarPair.
    /// Return None if the layout does not permit loading this as a value.
    pub(super) fn try_read_value_from_mplace(
        &self,
        mplace: MPlaceTy<'tcx>,
    ) -> EvalResult<'tcx, Option<Value>> {
        if mplace.extra != PlaceExtra::None {
            return Ok(None);
        }
        let (ptr, ptr_align) = mplace.to_scalar_ptr_align();

        if mplace.layout.size.bytes() == 0 {
            // Not all ZSTs have a layout we would handle below, so just short-circuit them
            // all here.
            self.memory.check_align(ptr, ptr_align)?;
            return Ok(Some(Value::Scalar(Scalar::zst().into())));
        }

        let ptr = ptr.to_ptr()?;
        match mplace.layout.abi {
            layout::Abi::Scalar(..) => {
                let scalar = self.memory.read_scalar(ptr, ptr_align, mplace.layout.size)?;
                Ok(Some(Value::Scalar(scalar)))
            }
            layout::Abi::ScalarPair(ref a, ref b) => {
                let (a, b) = (&a.value, &b.value);
                let (a_size, b_size) = (a.size(self), b.size(self));
                let a_ptr = ptr;
                let b_offset = a_size.abi_align(b.align(self));
                assert!(b_offset.bytes() > 0); // we later use the offset to test which field to use
                let b_ptr = ptr.offset(b_offset, self)?.into();
                let a_val = self.memory.read_scalar(a_ptr, ptr_align, a_size)?;
                let b_val = self.memory.read_scalar(b_ptr, ptr_align, b_size)?;
                Ok(Some(Value::ScalarPair(a_val, b_val)))
            }
            _ => Ok(None),
        }
    }

    /// Try returning an immediate value for the operand.
    /// If the layout does not permit loading this as a value, return where in memory
    /// we can find the data.
    /// Note that for a given layout, this operation will either always fail or always
    /// succeed!  Whether it succeeds depends on whether the layout can be represented
    /// in a `Value`, not on which data is stored there currently.
    pub(super) fn try_read_value(
        &self,
        src: OpTy<'tcx>,
    ) -> EvalResult<'tcx, Result<Value, MemPlace>> {
        Ok(match src.try_as_mplace() {
            Ok(mplace) => {
                if let Some(val) = self.try_read_value_from_mplace(mplace)? {
                    Ok(val)
                } else {
                    Err(*mplace)
                }
            },
            Err(val) => Ok(val),
        })
    }

    /// Read a value from a place, asserting that that is possible with the given layout.
    #[inline(always)]
    pub fn read_value(&self, op: OpTy<'tcx>) -> EvalResult<'tcx, ValTy<'tcx>> {
        if let Ok(value) = self.try_read_value(op)? {
            Ok(ValTy { value, layout: op.layout })
        } else {
            bug!("primitive read failed for type: {:?}", op.layout.ty);
        }
    }

    /// Read a scalar from a place
    pub fn read_scalar(&self, op: OpTy<'tcx>) -> EvalResult<'tcx, ScalarMaybeUndef> {
        match *self.read_value(op)? {
            Value::ScalarPair(..) => bug!("got ScalarPair for type: {:?}", op.layout.ty),
            Value::Scalar(val) => Ok(val),
        }
    }

    pub fn uninit_operand(&mut self, layout: TyLayout<'tcx>) -> EvalResult<'tcx, Operand> {
        // This decides which types we will use the Immediate optimization for, and hence should
        // match what `try_read_value` and `eval_place_to_op` support.
        if layout.is_zst() {
            return Ok(Operand::Immediate(Value::Scalar(Scalar::zst().into())));
        }

        Ok(match layout.abi {
            layout::Abi::Scalar(..) =>
                Operand::Immediate(Value::Scalar(ScalarMaybeUndef::Undef)),
            layout::Abi::ScalarPair(..) =>
                Operand::Immediate(Value::ScalarPair(
                    ScalarMaybeUndef::Undef,
                    ScalarMaybeUndef::Undef,
                )),
            _ => {
                trace!("Forcing allocation for local of type {:?}", layout.ty);
                Operand::Indirect(
                    *self.allocate(layout, MemoryKind::Stack)?
                )
            }
        })
    }

    /// Projection functions
    pub fn operand_field(
        &self,
        op: OpTy<'tcx>,
        field: u64,
    ) -> EvalResult<'tcx, OpTy<'tcx>> {
        let base = match op.try_as_mplace() {
            Ok(mplace) => {
                // The easy case
                let field = self.mplace_field(mplace, field)?;
                return Ok(field.into());
            },
            Err(value) => value
        };

        let field = field.try_into().unwrap();
        let field_layout = op.layout.field(self, field)?;
        if field_layout.size.bytes() == 0 {
            let val = Value::Scalar(Scalar::zst().into());
            return Ok(OpTy { op: Operand::Immediate(val), layout: field_layout });
        }
        let offset = op.layout.fields.offset(field);
        let value = match base {
            // the field covers the entire type
            _ if offset.bytes() == 0 && field_layout.size == op.layout.size => base,
            // extract fields from types with `ScalarPair` ABI
            Value::ScalarPair(a, b) => {
                let val = if offset.bytes() == 0 { a } else { b };
                Value::Scalar(val)
            },
            Value::Scalar(val) =>
                bug!("field access on non aggregate {:#?}, {:#?}", val, op.layout),
        };
        Ok(OpTy { op: Operand::Immediate(value), layout: field_layout })
    }

    pub(super) fn operand_downcast(
        &self,
        op: OpTy<'tcx>,
        variant: usize,
    ) -> EvalResult<'tcx, OpTy<'tcx>> {
        // Downcasts only change the layout
        Ok(match op.try_as_mplace() {
            Ok(mplace) => {
                self.mplace_downcast(mplace, variant)?.into()
            },
            Err(..) => {
                let layout = op.layout.for_variant(self, variant);
                OpTy { layout, ..op }
            }
        })
    }

    // Take an operand, representing a pointer, and dereference it -- that
    // will always be a MemPlace.
    pub(super) fn deref_operand(
        &self,
        src: OpTy<'tcx>,
    ) -> EvalResult<'tcx, MPlaceTy<'tcx>> {
        let val = self.read_value(src)?;
        trace!("deref to {} on {:?}", val.layout.ty, val);
        Ok(self.ref_to_mplace(val)?)
    }

    pub fn operand_projection(
        &self,
        base: OpTy<'tcx>,
        proj_elem: &mir::PlaceElem<'tcx>,
    ) -> EvalResult<'tcx, OpTy<'tcx>> {
        use rustc::mir::ProjectionElem::*;
        Ok(match *proj_elem {
            Field(field, _) => self.operand_field(base, field.index() as u64)?,
            Downcast(_, variant) => self.operand_downcast(base, variant)?,
            Deref => self.deref_operand(base)?.into(),
            // The rest should only occur as mplace, we do not use Immediates for types
            // allowing such operations.  This matches place_projection forcing an allocation.
            Subslice { .. } | ConstantIndex { .. } | Index(_) => {
                let mplace = base.to_mem_place();
                self.mplace_projection(mplace, proj_elem)?.into()
            }
        })
    }

    // Evaluate a place with the goal of reading from it.  This lets us sometimes
    // avoid allocations.  If you already know the layout, you can pass it in
    // to avoid looking it up again.
    fn eval_place_to_op(
        &mut self,
        mir_place: &mir::Place<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> EvalResult<'tcx, OpTy<'tcx>> {
        use rustc::mir::Place::*;
        Ok(match *mir_place {
            Local(mir::RETURN_PLACE) => return err!(ReadFromReturnPointer),
            Local(local) => {
                let op = *self.frame().locals[local].access()?;
                let layout = from_known_layout(layout,
                    || self.layout_of_local(self.cur_frame(), local))?;
                OpTy { op, layout }
            },

            Projection(ref proj) => {
                let op = self.eval_place_to_op(&proj.base, None)?;
                self.operand_projection(op, &proj.elem)?
            }

            // Everything else is an mplace, so we just call `eval_place`.
            // Note that getting an mplace for a static aways requires `&mut`,
            // so this does not "cost" us anything in terms if mutability.
            Promoted(_) | Static(_) => {
                let place = self.eval_place(mir_place)?;
                place.to_mem_place().into()
            }
        })
    }

    /// Evaluate the operand, returning a place where you can then find the data.
    /// if you already know the layout, you can save two some table lookups
    /// by passing it in here.
    pub fn eval_operand(
        &mut self,
        mir_op: &mir::Operand<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> EvalResult<'tcx, OpTy<'tcx>> {
        use rustc::mir::Operand::*;
        let op = match *mir_op {
            // FIXME: do some more logic on `move` to invalidate the old location
            Copy(ref place) |
            Move(ref place) =>
                self.eval_place_to_op(place, layout)?,

            Constant(ref constant) => {
                let layout = from_known_layout(layout, || {
                    let ty = self.monomorphize(mir_op.ty(self.mir(), *self.tcx), self.substs());
                    self.layout_of(ty)
                })?;
                let op = self.const_value_to_op(constant.literal.val)?;
                OpTy { op, layout }
            }
        };
        trace!("{:?}: {:?}", mir_op, *op);
        Ok(op)
    }

    /// Evaluate a bunch of operands at once
    pub(crate) fn eval_operands(
        &mut self,
        ops: &[mir::Operand<'tcx>],
    ) -> EvalResult<'tcx, Vec<OpTy<'tcx>>> {
        ops.into_iter()
            .map(|op| self.eval_operand(op, None))
            .collect()
    }

    // Also used e.g. when miri runs into a constant.
    // Unfortunately, this needs an `&mut` to be able to allocate a copy of a `ByRef`
    // constant.  This bleeds up to `eval_operand` needing `&mut`.
    pub fn const_value_to_op(
        &mut self,
        val: ConstValue<'tcx>,
    ) -> EvalResult<'tcx, Operand> {
        match val {
            ConstValue::Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                self.global_to_op(GlobalId {
                    instance,
                    promoted: None,
                })
            }
            ConstValue::ByRef(alloc, offset) => {
                // FIXME: Allocate new AllocId for all constants inside
                let id = self.memory.allocate_value(alloc.clone(), MemoryKind::Stack)?;
                Ok(Operand::from_ptr(Pointer::new(id, offset), alloc.align))
            },
            ConstValue::ScalarPair(a, b) =>
                Ok(Operand::Immediate(Value::ScalarPair(a.into(), b))),
            ConstValue::Scalar(x) =>
                Ok(Operand::Immediate(Value::Scalar(x.into()))),
        }
    }

    pub(super) fn global_to_op(&mut self, gid: GlobalId<'tcx>) -> EvalResult<'tcx, Operand> {
        let cv = self.const_eval(gid)?;
        self.const_value_to_op(cv.val)
    }

    /// We cannot do self.read_value(self.eval_operand) due to eval_operand taking &mut self,
    /// so this helps avoid unnecessary let.
    #[inline]
    pub fn eval_operand_and_read_value(
        &mut self,
        op: &mir::Operand<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> EvalResult<'tcx, ValTy<'tcx>> {
        let op = self.eval_operand(op, layout)?;
        self.read_value(op)
    }

    /// reads a tag and produces the corresponding variant index
    pub fn read_discriminant_as_variant_index(
        &self,
        rval: OpTy<'tcx>,
    ) -> EvalResult<'tcx, usize> {
        match rval.layout.variants {
            layout::Variants::Single { index } => Ok(index),
            layout::Variants::Tagged { .. } => {
                let discr_val = self.read_discriminant_value(rval)?;
                rval.layout.ty
                    .ty_adt_def()
                    .expect("tagged layout for non adt")
                    .discriminants(self.tcx.tcx)
                    .position(|var| var.val == discr_val)
                    .ok_or_else(|| EvalErrorKind::InvalidDiscriminant.into())
            }
            layout::Variants::NicheFilling { .. } => {
                let discr_val = self.read_discriminant_value(rval)?;
                assert_eq!(discr_val as usize as u128, discr_val);
                Ok(discr_val as usize)
            },
        }
    }

    pub fn read_discriminant_value(
        &self,
        rval: OpTy<'tcx>,
    ) -> EvalResult<'tcx, u128> {
        trace!("read_discriminant_value {:#?}", rval.layout);
        if rval.layout.abi == layout::Abi::Uninhabited {
            return err!(Unreachable);
        }

        match rval.layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = rval.layout.ty.ty_adt_def().map_or(
                    index as u128,
                    |def| def.discriminant_for_variant(*self.tcx, index).val);
                return Ok(discr_val);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }
        let discr_op = self.operand_field(rval, 0)?;
        let discr_val = self.read_value(discr_op)?;
        trace!("discr value: {:?}", discr_val);
        let raw_discr = discr_val.to_scalar()?;
        Ok(match rval.layout.variants {
            layout::Variants::Single { .. } => bug!(),
            // FIXME: We should catch invalid discriminants here!
            layout::Variants::Tagged { .. } => {
                if discr_val.layout.ty.is_signed() {
                    let i = raw_discr.to_bits(discr_val.layout.size)? as i128;
                    // going from layout tag type to typeck discriminant type
                    // requires first sign extending with the layout discriminant
                    let shift = 128 - discr_val.layout.size.bits();
                    let sexted = (i << shift) >> shift;
                    // and then zeroing with the typeck discriminant type
                    let discr_ty = rval.layout.ty
                        .ty_adt_def().expect("tagged layout corresponds to adt")
                        .repr
                        .discr_type();
                    let discr_ty = layout::Integer::from_attr(self.tcx.tcx, discr_ty);
                    let shift = 128 - discr_ty.size().bits();
                    let truncatee = sexted as u128;
                    (truncatee << shift) >> shift
                } else {
                    raw_discr.to_bits(discr_val.layout.size)?
                }
            },
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let variants_start = *niche_variants.start() as u128;
                let variants_end = *niche_variants.end() as u128;
                match raw_discr {
                    Scalar::Ptr(_) => {
                        assert!(niche_start == 0);
                        assert!(variants_start == variants_end);
                        dataful_variant as u128
                    },
                    Scalar::Bits { bits: raw_discr, size } => {
                        assert_eq!(size as u64, discr_val.layout.size.bytes());
                        let discr = raw_discr.wrapping_sub(niche_start)
                            .wrapping_add(variants_start);
                        if variants_start <= discr && discr <= variants_end {
                            discr
                        } else {
                            dataful_variant as u128
                        }
                    },
                }
            }
        })
    }

}
