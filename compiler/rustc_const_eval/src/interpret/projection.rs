//! This file implements "place projections"; basically a symmetric API for 3 types: MPlaceTy, OpTy, PlaceTy.
//!
//! OpTy and PlaceTy generally work by "let's see if we are actually an MPlaceTy, and do something custom if not".
//! For PlaceTy, the custom thing is basically always to call `force_allocation` and then use the MPlaceTy logic anyway.
//! For OpTy, the custom thing on field pojections has to be pretty clever (since `Operand::Immediate` can have fields),
//! but for array/slice operations it only has to worry about `Operand::Uninit`. That makes the value part trivial,
//! but we still need to do bounds checking and adjust the layout. To not duplicate that with MPlaceTy, we actually
//! implement the logic on OpTy, and MPlaceTy calls that.

use rustc_middle::mir;
use rustc_middle::ty;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::Ty;
use rustc_middle::ty::TyCtxt;
use rustc_target::abi::HasDataLayout;
use rustc_target::abi::Size;
use rustc_target::abi::{self, VariantIdx};

use super::MPlaceTy;
use super::{InterpCx, InterpResult, Machine, MemPlaceMeta, OpTy, Provenance, Scalar};

/// A thing that we can project into, and that has a layout.
pub trait Projectable<'mir, 'tcx: 'mir, Prov: Provenance>: Sized {
    /// Get the layout.
    fn layout(&self) -> TyAndLayout<'tcx>;

    /// Get the metadata of a wide value.
    fn meta<M: Machine<'mir, 'tcx, Provenance = Prov>>(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, MemPlaceMeta<M::Provenance>>;

    fn len<M: Machine<'mir, 'tcx, Provenance = Prov>>(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, u64> {
        self.meta(ecx)?.len(self.layout(), ecx)
    }

    /// Offset the value by the given amount, replacing the layout and metadata.
    fn offset_with_meta(
        &self,
        offset: Size,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self>;

    fn offset(
        &self,
        offset: Size,
        layout: TyAndLayout<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self> {
        assert!(layout.is_sized());
        self.offset_with_meta(offset, MemPlaceMeta::None, layout, cx)
    }

    fn transmute(
        &self,
        layout: TyAndLayout<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self> {
        assert_eq!(self.layout().size, layout.size);
        self.offset_with_meta(Size::ZERO, MemPlaceMeta::None, layout, cx)
    }

    /// Convert this to an `OpTy`. This might be an irreversible transformation, but is useful for
    /// reading from this thing.
    fn to_op<M: Machine<'mir, 'tcx, Provenance = Prov>>(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>>;
}

// FIXME: Working around https://github.com/rust-lang/rust/issues/54385
impl<'mir, 'tcx: 'mir, Prov, M> InterpCx<'mir, 'tcx, M>
where
    Prov: Provenance + 'static,
    M: Machine<'mir, 'tcx, Provenance = Prov>,
{
    /// Offset a pointer to project to a field of a struct/union. Unlike `place_field`, this is
    /// always possible without allocating, so it can take `&self`. Also return the field's layout.
    /// This supports both struct and array fields, but not slices!
    ///
    /// This also works for arrays, but then the `usize` index type is restricting.
    /// For indexing into arrays, use `mplace_index`.
    pub fn project_field<P: Projectable<'mir, 'tcx, M::Provenance>>(
        &self,
        base: &P,
        field: usize,
    ) -> InterpResult<'tcx, P> {
        // Slices nominally have length 0, so they will panic somewhere in `fields.offset`.
        debug_assert!(
            !matches!(base.layout().ty.kind(), ty::Slice(..)),
            "`field` projection called on a slice -- call `index` projection instead"
        );
        let offset = base.layout().fields.offset(field);
        let field_layout = base.layout().field(self, field);

        // Offset may need adjustment for unsized fields.
        let (meta, offset) = if field_layout.is_unsized() {
            if base.layout().is_sized() {
                // An unsized field of a sized type? Sure...
                // But const-prop actually feeds us such nonsense MIR!
                throw_inval!(ConstPropNonsense);
            }
            let base_meta = base.meta(self)?;
            // Re-use parent metadata to determine dynamic field layout.
            // With custom DSTS, this *will* execute user-defined code, but the same
            // happens at run-time so that's okay.
            match self.size_and_align_of(&base_meta, &field_layout)? {
                Some((_, align)) => (base_meta, offset.align_to(align)),
                None => {
                    // For unsized types with an extern type tail we perform no adjustments.
                    // NOTE: keep this in sync with `PlaceRef::project_field` in the codegen backend.
                    assert!(matches!(base_meta, MemPlaceMeta::None));
                    (base_meta, offset)
                }
            }
        } else {
            // base_meta could be present; we might be accessing a sized field of an unsized
            // struct.
            (MemPlaceMeta::None, offset)
        };

        base.offset_with_meta(offset, meta, field_layout, self)
    }

    /// Downcasting to an enum variant.
    pub fn project_downcast<P: Projectable<'mir, 'tcx, M::Provenance>>(
        &self,
        base: &P,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, P> {
        assert!(!base.meta(self)?.has_meta());
        // Downcasts only change the layout.
        // (In particular, no check about whether this is even the active variant -- that's by design,
        // see https://github.com/rust-lang/rust/issues/93688#issuecomment-1032929496.)
        // So we just "offset" by 0.
        let layout = base.layout().for_variant(self, variant);
        if layout.abi.is_uninhabited() {
            // `read_discriminant` should have excluded uninhabited variants... but ConstProp calls
            // us on dead code.
            throw_inval!(ConstPropNonsense)
        }
        // This cannot be `transmute` as variants *can* have a smaller size than the entire enum.
        base.offset(Size::ZERO, layout, self)
    }

    /// Compute the offset and field layout for accessing the given index.
    pub fn project_index<P: Projectable<'mir, 'tcx, M::Provenance>>(
        &self,
        base: &P,
        index: u64,
    ) -> InterpResult<'tcx, P> {
        // Not using the layout method because we want to compute on u64
        let (offset, field_layout) = match base.layout().fields {
            abi::FieldsShape::Array { stride, count: _ } => {
                // `count` is nonsense for slices, use the dynamic length instead.
                let len = base.len(self)?;
                if index >= len {
                    // This can only be reached in ConstProp and non-rustc-MIR.
                    throw_ub!(BoundsCheckFailed { len, index });
                }
                let offset = stride * index; // `Size` multiplication
                // All fields have the same layout.
                let field_layout = base.layout().field(self, 0);
                (offset, field_layout)
            }
            _ => span_bug!(
                self.cur_span(),
                "`mplace_index` called on non-array type {:?}",
                base.layout().ty
            ),
        };

        base.offset(offset, field_layout, self)
    }

    fn project_constant_index<P: Projectable<'mir, 'tcx, M::Provenance>>(
        &self,
        base: &P,
        offset: u64,
        min_length: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, P> {
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

        self.project_index(base, index)
    }

    /// Iterates over all fields of an array. Much more efficient than doing the
    /// same by repeatedly calling `operand_index`.
    pub fn project_array_fields<'a, P: Projectable<'mir, 'tcx, M::Provenance>>(
        &self,
        base: &'a P,
    ) -> InterpResult<'tcx, impl Iterator<Item = InterpResult<'tcx, P>> + 'a>
    where
        'tcx: 'a,
    {
        let abi::FieldsShape::Array { stride, .. } = base.layout().fields else {
            span_bug!(self.cur_span(), "operand_array_fields: expected an array layout");
        };
        let len = base.len(self)?;
        let field_layout = base.layout().field(self, 0);
        let tcx: TyCtxt<'tcx> = *self.tcx;
        // `Size` multiplication
        Ok((0..len).map(move |i| {
            base.offset_with_meta(stride * i, MemPlaceMeta::None, field_layout, &tcx)
        }))
    }

    /// Subslicing
    fn project_subslice<P: Projectable<'mir, 'tcx, M::Provenance>>(
        &self,
        base: &P,
        from: u64,
        to: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, P> {
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
        let from_offset = match base.layout().fields {
            abi::FieldsShape::Array { stride, .. } => stride * from, // `Size` multiplication is checked
            _ => {
                span_bug!(
                    self.cur_span(),
                    "unexpected layout of index access: {:#?}",
                    base.layout()
                )
            }
        };

        // Compute meta and new layout
        let inner_len = actual_to.checked_sub(from).unwrap();
        let (meta, ty) = match base.layout().ty.kind() {
            // It is not nice to match on the type, but that seems to be the only way to
            // implement this.
            ty::Array(inner, _) => {
                (MemPlaceMeta::None, Ty::new_array(self.tcx.tcx, *inner, inner_len))
            }
            ty::Slice(..) => {
                let len = Scalar::from_target_usize(inner_len, self);
                (MemPlaceMeta::Meta(len), base.layout().ty)
            }
            _ => {
                span_bug!(
                    self.cur_span(),
                    "cannot subslice non-array type: `{:?}`",
                    base.layout().ty
                )
            }
        };
        let layout = self.layout_of(ty)?;

        base.offset_with_meta(from_offset, meta, layout, self)
    }

    /// Applying a general projection
    #[instrument(skip(self), level = "trace")]
    pub fn project<P>(&self, base: &P, proj_elem: mir::PlaceElem<'tcx>) -> InterpResult<'tcx, P>
    where
        P: Projectable<'mir, 'tcx, M::Provenance>
            + From<MPlaceTy<'tcx, M::Provenance>>
            + std::fmt::Debug,
    {
        use rustc_middle::mir::ProjectionElem::*;
        Ok(match proj_elem {
            OpaqueCast(ty) => base.transmute(self.layout_of(ty)?, self)?,
            Field(field, _) => self.project_field(base, field.index())?,
            Downcast(_, variant) => self.project_downcast(base, variant)?,
            Deref => self.deref_operand(&base.to_op(self)?)?.into(),
            Index(local) => {
                let layout = self.layout_of(self.tcx.types.usize)?;
                let n = self.local_to_op(self.frame(), local, Some(layout))?;
                let n = self.read_target_usize(&n)?;
                self.project_index(base, n)?
            }
            ConstantIndex { offset, min_length, from_end } => {
                self.project_constant_index(base, offset, min_length, from_end)?
            }
            Subslice { from, to, from_end } => self.project_subslice(base, from, to, from_end)?,
        })
    }
}
