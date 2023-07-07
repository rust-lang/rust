//! This file implements "place projections"; basically a symmetric API for 3 types: MPlaceTy, OpTy, PlaceTy.
//!
//! OpTy and PlaceTy generally work by "let's see if we are actually an MPlaceTy, and do something custom if not".
//! For PlaceTy, the custom thing is basically always to call `force_allocation` and then use the MPlaceTy logic anyway.
//! For OpTy, the custom thing on field pojections has to be pretty clever (since `Operand::Immediate` can have fields),
//! but for array/slice operations it only has to worry about `Operand::Uninit`. That makes the value part trivial,
//! but we still need to do bounds checking and adjust the layout. To not duplicate that with MPlaceTy, we actually
//! implement the logic on OpTy, and MPlaceTy calls that.

use either::{Left, Right};

use rustc_middle::mir;
use rustc_middle::ty;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::Ty;
use rustc_target::abi::{self, Abi, VariantIdx};

use super::{
    ImmTy, Immediate, InterpCx, InterpResult, MPlaceTy, Machine, MemPlaceMeta, OpTy, PlaceTy,
    Provenance, Scalar,
};

// FIXME: Working around https://github.com/rust-lang/rust/issues/54385
impl<'mir, 'tcx: 'mir, Prov, M> InterpCx<'mir, 'tcx, M>
where
    Prov: Provenance + 'static,
    M: Machine<'mir, 'tcx, Provenance = Prov>,
{
    //# Field access

    /// Offset a pointer to project to a field of a struct/union. Unlike `place_field`, this is
    /// always possible without allocating, so it can take `&self`. Also return the field's layout.
    /// This supports both struct and array fields.
    ///
    /// This also works for arrays, but then the `usize` index type is restricting.
    /// For indexing into arrays, use `mplace_index`.
    pub fn mplace_field(
        &self,
        base: &MPlaceTy<'tcx, M::Provenance>,
        field: usize,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let offset = base.layout.fields.offset(field);
        let field_layout = base.layout.field(self, field);

        // Offset may need adjustment for unsized fields.
        let (meta, offset) = if field_layout.is_unsized() {
            // Re-use parent metadata to determine dynamic field layout.
            // With custom DSTS, this *will* execute user-defined code, but the same
            // happens at run-time so that's okay.
            match self.size_and_align_of(&base.meta, &field_layout)? {
                Some((_, align)) => (base.meta, offset.align_to(align)),
                None => {
                    // For unsized types with an extern type tail we perform no adjustments.
                    // NOTE: keep this in sync with `PlaceRef::project_field` in the codegen backend.
                    assert!(matches!(base.meta, MemPlaceMeta::None));
                    (base.meta, offset)
                }
            }
        } else {
            // base.meta could be present; we might be accessing a sized field of an unsized
            // struct.
            (MemPlaceMeta::None, offset)
        };

        // We do not look at `base.layout.align` nor `field_layout.align`, unlike
        // codegen -- mostly to see if we can get away with that
        base.offset_with_meta(offset, meta, field_layout, self)
    }

    /// Gets the place of a field inside the place, and also the field's type.
    /// Just a convenience function, but used quite a bit.
    /// This is the only projection that might have a side-effect: We cannot project
    /// into the field of a local `ScalarPair`, we have to first allocate it.
    pub fn place_field(
        &mut self,
        base: &PlaceTy<'tcx, M::Provenance>,
        field: usize,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        // FIXME: We could try to be smarter and avoid allocation for fields that span the
        // entire place.
        let base = self.force_allocation(base)?;
        Ok(self.mplace_field(&base, field)?.into())
    }

    pub fn operand_field(
        &self,
        base: &OpTy<'tcx, M::Provenance>,
        field: usize,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        let base = match base.as_mplace_or_imm() {
            Left(ref mplace) => {
                // We can reuse the mplace field computation logic for indirect operands.
                let field = self.mplace_field(mplace, field)?;
                return Ok(field.into());
            }
            Right(value) => value,
        };

        let field_layout = base.layout.field(self, field);
        let offset = base.layout.fields.offset(field);
        // This makes several assumptions about what layouts we will encounter; we match what
        // codegen does as good as we can (see `extract_field` in `rustc_codegen_ssa/src/mir/operand.rs`).
        let field_val: Immediate<_> = match (*base, base.layout.abi) {
            // if the entire value is uninit, then so is the field (can happen in ConstProp)
            (Immediate::Uninit, _) => Immediate::Uninit,
            // the field contains no information, can be left uninit
            _ if field_layout.is_zst() => Immediate::Uninit,
            // the field covers the entire type
            _ if field_layout.size == base.layout.size => {
                assert!(match (base.layout.abi, field_layout.abi) {
                    (Abi::Scalar(..), Abi::Scalar(..)) => true,
                    (Abi::ScalarPair(..), Abi::ScalarPair(..)) => true,
                    _ => false,
                });
                assert!(offset.bytes() == 0);
                *base
            }
            // extract fields from types with `ScalarPair` ABI
            (Immediate::ScalarPair(a_val, b_val), Abi::ScalarPair(a, b)) => {
                assert!(matches!(field_layout.abi, Abi::Scalar(..)));
                Immediate::from(if offset.bytes() == 0 {
                    debug_assert_eq!(field_layout.size, a.size(self));
                    a_val
                } else {
                    debug_assert_eq!(offset, a.size(self).align_to(b.align(self).abi));
                    debug_assert_eq!(field_layout.size, b.size(self));
                    b_val
                })
            }
            // everything else is a bug
            _ => span_bug!(
                self.cur_span(),
                "invalid field access on immediate {}, layout {:#?}",
                base,
                base.layout
            ),
        };

        Ok(ImmTy::from_immediate(field_val, field_layout).into())
    }

    //# Downcasting

    pub fn mplace_downcast(
        &self,
        base: &MPlaceTy<'tcx, M::Provenance>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        // Downcasts only change the layout.
        // (In particular, no check about whether this is even the active variant -- that's by design,
        // see https://github.com/rust-lang/rust/issues/93688#issuecomment-1032929496.)
        assert!(!base.meta.has_meta());
        let mut base = *base;
        base.layout = base.layout.for_variant(self, variant);
        Ok(base)
    }

    pub fn place_downcast(
        &self,
        base: &PlaceTy<'tcx, M::Provenance>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        // Downcast just changes the layout
        let mut base = base.clone();
        base.layout = base.layout.for_variant(self, variant);
        Ok(base)
    }

    pub fn operand_downcast(
        &self,
        base: &OpTy<'tcx, M::Provenance>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // Downcast just changes the layout
        let mut base = base.clone();
        base.layout = base.layout.for_variant(self, variant);
        Ok(base)
    }

    //# Slice indexing

    #[inline(always)]
    pub fn operand_index(
        &self,
        base: &OpTy<'tcx, M::Provenance>,
        index: u64,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // Not using the layout method because we want to compute on u64
        match base.layout.fields {
            abi::FieldsShape::Array { stride, count: _ } => {
                // `count` is nonsense for slices, use the dynamic length instead.
                let len = base.len(self)?;
                if index >= len {
                    // This can only be reached in ConstProp and non-rustc-MIR.
                    throw_ub!(BoundsCheckFailed { len, index });
                }
                let offset = stride * index; // `Size` multiplication
                // All fields have the same layout.
                let field_layout = base.layout.field(self, 0);
                base.offset(offset, field_layout, self)
            }
            _ => span_bug!(
                self.cur_span(),
                "`mplace_index` called on non-array type {:?}",
                base.layout.ty
            ),
        }
    }

    /// Iterates over all fields of an array. Much more efficient than doing the
    /// same by repeatedly calling `operand_index`.
    pub fn operand_array_fields<'a>(
        &self,
        base: &'a OpTy<'tcx, Prov>,
    ) -> InterpResult<'tcx, impl Iterator<Item = InterpResult<'tcx, OpTy<'tcx, Prov>>> + 'a> {
        let len = base.len(self)?; // also asserts that we have a type where this makes sense
        let abi::FieldsShape::Array { stride, .. } = base.layout.fields else {
            span_bug!(self.cur_span(), "operand_array_fields: expected an array layout");
        };
        let field_layout = base.layout.field(self, 0);
        let dl = &self.tcx.data_layout;
        // `Size` multiplication
        Ok((0..len).map(move |i| base.offset(stride * i, field_layout, dl)))
    }

    /// Index into an array.
    pub fn mplace_index(
        &self,
        base: &MPlaceTy<'tcx, M::Provenance>,
        index: u64,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        Ok(self.operand_index(&base.into(), index)?.assert_mem_place())
    }

    pub fn place_index(
        &mut self,
        base: &PlaceTy<'tcx, M::Provenance>,
        index: u64,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        // There's not a lot we can do here, since we cannot have a place to a part of a local. If
        // we are accessing the only element of a 1-element array, it's still the entire local...
        // that doesn't seem worth it.
        let base = self.force_allocation(base)?;
        Ok(self.mplace_index(&base, index)?.into())
    }

    //# ConstantIndex support

    fn operand_constant_index(
        &self,
        base: &OpTy<'tcx, M::Provenance>,
        offset: u64,
        min_length: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        let n = base.len(self)?;
        if n < min_length {
            // This can only be reached in ConstProp and non-rustc-MIR.
            throw_ub!(BoundsCheckFailed { len: min_length, index: n });
        }

        let index = if from_end {
            assert!(0 < offset && offset <= min_length);
            n.checked_sub(offset).unwrap()
        } else {
            assert!(offset < min_length);
            offset
        };

        self.operand_index(base, index)
    }

    fn place_constant_index(
        &mut self,
        base: &PlaceTy<'tcx, M::Provenance>,
        offset: u64,
        min_length: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        let base = self.force_allocation(base)?;
        Ok(self
            .operand_constant_index(&base.into(), offset, min_length, from_end)?
            .assert_mem_place()
            .into())
    }

    //# Subslicing

    fn operand_subslice(
        &self,
        base: &OpTy<'tcx, M::Provenance>,
        from: u64,
        to: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        let len = base.len(self)?; // also asserts that we have a type where this makes sense
        let actual_to = if from_end {
            if from.checked_add(to).map_or(true, |to| to > len) {
                // This can only be reached in ConstProp and non-rustc-MIR.
                throw_ub!(BoundsCheckFailed { len: len, index: from.saturating_add(to) });
            }
            len.checked_sub(to).unwrap()
        } else {
            to
        };

        // Not using layout method because that works with usize, and does not work with slices
        // (that have count 0 in their layout).
        let from_offset = match base.layout.fields {
            abi::FieldsShape::Array { stride, .. } => stride * from, // `Size` multiplication is checked
            _ => {
                span_bug!(self.cur_span(), "unexpected layout of index access: {:#?}", base.layout)
            }
        };

        // Compute meta and new layout
        let inner_len = actual_to.checked_sub(from).unwrap();
        let (meta, ty) = match base.layout.ty.kind() {
            // It is not nice to match on the type, but that seems to be the only way to
            // implement this.
            ty::Array(inner, _) => {
                (MemPlaceMeta::None, Ty::new_array(self.tcx.tcx, *inner, inner_len))
            }
            ty::Slice(..) => {
                let len = Scalar::from_target_usize(inner_len, self);
                (MemPlaceMeta::Meta(len), base.layout.ty)
            }
            _ => {
                span_bug!(self.cur_span(), "cannot subslice non-array type: `{:?}`", base.layout.ty)
            }
        };
        let layout = self.layout_of(ty)?;
        base.offset_with_meta(from_offset, meta, layout, self)
    }

    pub fn place_subslice(
        &mut self,
        base: &PlaceTy<'tcx, M::Provenance>,
        from: u64,
        to: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        let base = self.force_allocation(base)?;
        Ok(self.operand_subslice(&base.into(), from, to, from_end)?.assert_mem_place().into())
    }

    //# Applying a general projection

    /// Projects into a place.
    #[instrument(skip(self), level = "trace")]
    pub fn place_projection(
        &mut self,
        base: &PlaceTy<'tcx, M::Provenance>,
        proj_elem: mir::PlaceElem<'tcx>,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::ProjectionElem::*;
        Ok(match proj_elem {
            OpaqueCast(ty) => {
                let mut place = base.clone();
                place.layout = self.layout_of(ty)?;
                place
            }
            Field(field, _) => self.place_field(base, field.index())?,
            Downcast(_, variant) => self.place_downcast(base, variant)?,
            Deref => self.deref_operand(&self.place_to_op(base)?)?.into(),
            Index(local) => {
                let layout = self.layout_of(self.tcx.types.usize)?;
                let n = self.local_to_op(self.frame(), local, Some(layout))?;
                let n = self.read_target_usize(&n)?;
                self.place_index(base, n)?
            }
            ConstantIndex { offset, min_length, from_end } => {
                self.place_constant_index(base, offset, min_length, from_end)?
            }
            Subslice { from, to, from_end } => self.place_subslice(base, from, to, from_end)?,
        })
    }

    #[instrument(skip(self), level = "trace")]
    pub fn operand_projection(
        &self,
        base: &OpTy<'tcx, M::Provenance>,
        proj_elem: mir::PlaceElem<'tcx>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::ProjectionElem::*;
        Ok(match proj_elem {
            OpaqueCast(ty) => {
                let mut op = base.clone();
                op.layout = self.layout_of(ty)?;
                op
            }
            Field(field, _) => self.operand_field(base, field.index())?,
            Downcast(_, variant) => self.operand_downcast(base, variant)?,
            Deref => self.deref_operand(base)?.into(),
            Index(local) => {
                let layout = self.layout_of(self.tcx.types.usize)?;
                let n = self.local_to_op(self.frame(), local, Some(layout))?;
                let n = self.read_target_usize(&n)?;
                self.operand_index(base, n)?
            }
            ConstantIndex { offset, min_length, from_end } => {
                self.operand_constant_index(base, offset, min_length, from_end)?
            }
            Subslice { from, to, from_end } => self.operand_subslice(base, from, to, from_end)?,
        })
    }
}
