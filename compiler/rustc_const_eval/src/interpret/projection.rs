//! This file implements "place projections"; basically a symmetric API for 3 types: MPlaceTy, OpTy, PlaceTy.
//!
//! OpTy and PlaceTy generally work by "let's see if we are actually an MPlaceTy, and do something custom if not".
//! For PlaceTy, the custom thing is basically always to call `force_allocation` and then use the MPlaceTy logic anyway.
//! For OpTy, the custom thing on field pojections has to be pretty clever (since `Operand::Immediate` can have fields),
//! but for array/slice operations it only has to worry about `Operand::Uninit`. That makes the value part trivial,
//! but we still need to do bounds checking and adjust the layout. To not duplicate that with MPlaceTy, we actually
//! implement the logic on OpTy, and MPlaceTy calls that.

use std::marker::PhantomData;
use std::ops::Range;

use rustc_abi::{self as abi, FieldIdx, Size, VariantIdx};
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::{bug, mir, span_bug, ty};
use tracing::{debug, instrument};

use super::{
    InterpCx, InterpResult, MPlaceTy, Machine, MemPlaceMeta, OpTy, Provenance, Scalar, err_ub,
    interp_ok, throw_ub, throw_unsup,
};

/// Describes the constraints placed on offset-projections.
#[derive(Copy, Clone, Debug)]
pub enum OffsetMode {
    /// The offset has to be inbounds, like `ptr::offset`.
    Inbounds,
    /// No constraints, just wrap around the edge of the address space.
    Wrapping,
}

/// A thing that we can project into, and that has a layout.
pub trait Projectable<'tcx, Prov: Provenance>: Sized + std::fmt::Debug {
    /// Get the layout.
    fn layout(&self) -> TyAndLayout<'tcx>;

    /// Get the metadata of a wide value.
    fn meta(&self) -> MemPlaceMeta<Prov>;

    /// Get the length of a slice/string/array stored here.
    fn len<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, u64> {
        let layout = self.layout();
        if layout.is_unsized() {
            // We need to consult `meta` metadata
            match layout.ty.kind() {
                ty::Slice(..) | ty::Str => self.meta().unwrap_meta().to_target_usize(ecx),
                _ => bug!("len not supported on unsized type {:?}", layout.ty),
            }
        } else {
            // Go through the layout. There are lots of types that support a length,
            // e.g., SIMD types. (But not all repr(simd) types even have FieldsShape::Array!)
            match layout.fields {
                abi::FieldsShape::Array { count, .. } => interp_ok(count),
                _ => bug!("len not supported on sized type {:?}", layout.ty),
            }
        }
    }

    /// Offset the value by the given amount, replacing the layout and metadata.
    fn offset_with_meta<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        offset: Size,
        mode: OffsetMode,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self>;

    fn offset<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        offset: Size,
        layout: TyAndLayout<'tcx>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self> {
        assert!(layout.is_sized());
        // We sometimes do pointer arithmetic with this function, disregarding the source type.
        // So we don't check the sizes here.
        self.offset_with_meta(offset, OffsetMode::Inbounds, MemPlaceMeta::None, layout, ecx)
    }

    /// This does an offset-by-zero, which is effectively a transmute. Note however that
    /// not all transmutes are supported by all projectables -- specifically, if this is an
    /// `OpTy` or `ImmTy`, the new layout must have almost the same ABI as the old one
    /// (only changing the `valid_range` is allowed and turning integers into pointers).
    fn transmute<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        layout: TyAndLayout<'tcx>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self> {
        assert!(self.layout().is_sized() && layout.is_sized());
        assert_eq!(self.layout().size, layout.size);
        self.offset_with_meta(Size::ZERO, OffsetMode::Wrapping, MemPlaceMeta::None, layout, ecx)
    }

    /// Convert this to an `OpTy`. This might be an irreversible transformation, but is useful for
    /// reading from this thing.
    fn to_op<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>>;
}

/// A type representing iteration over the elements of an array.
pub struct ArrayIterator<'a, 'tcx, Prov: Provenance, P: Projectable<'tcx, Prov>> {
    base: &'a P,
    range: Range<u64>,
    stride: Size,
    field_layout: TyAndLayout<'tcx>,
    _phantom: PhantomData<Prov>, // otherwise it says `Prov` is never used...
}

impl<'a, 'tcx, Prov: Provenance, P: Projectable<'tcx, Prov>> ArrayIterator<'a, 'tcx, Prov, P> {
    /// Should be the same `ecx` on each call, and match the one used to create the iterator.
    pub fn next<M: Machine<'tcx, Provenance = Prov>>(
        &mut self,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Option<(u64, P)>> {
        let Some(idx) = self.range.next() else { return interp_ok(None) };
        // We use `Wrapping` here since the offset has already been checked when the iterator was created.
        interp_ok(Some((
            idx,
            self.base.offset_with_meta(
                self.stride * idx,
                OffsetMode::Wrapping,
                MemPlaceMeta::None,
                self.field_layout,
                ecx,
            )?,
        )))
    }
}

// FIXME: Working around https://github.com/rust-lang/rust/issues/54385
impl<'tcx, Prov, M> InterpCx<'tcx, M>
where
    Prov: Provenance,
    M: Machine<'tcx, Provenance = Prov>,
{
    /// Offset a pointer to project to a field of a struct/union. Unlike `place_field`, this is
    /// always possible without allocating, so it can take `&self`. Also return the field's layout.
    /// This supports both struct and array fields, but not slices!
    ///
    /// This also works for arrays, but then the `FieldIdx` index type is restricting.
    /// For indexing into arrays, use [`Self::project_index`].
    pub fn project_field<P: Projectable<'tcx, M::Provenance>>(
        &self,
        base: &P,
        field: FieldIdx,
    ) -> InterpResult<'tcx, P> {
        // Slices nominally have length 0, so they will panic somewhere in `fields.offset`.
        debug_assert!(
            !matches!(base.layout().ty.kind(), ty::Slice(..)),
            "`field` projection called on a slice -- call `index` projection instead"
        );
        let offset = base.layout().fields.offset(field.as_usize());
        // Computing the layout does normalization, so we get a normalized type out of this
        // even if the field type is non-normalized (possible e.g. via associated types).
        let field_layout = base.layout().field(self, field.as_usize());

        // Offset may need adjustment for unsized fields.
        let (meta, offset) = if field_layout.is_unsized() {
            assert!(!base.layout().is_sized());
            let base_meta = base.meta();
            // Re-use parent metadata to determine dynamic field layout.
            // With custom DSTS, this *will* execute user-defined code, but the same
            // happens at run-time so that's okay.
            match self.size_and_align_of(&base_meta, &field_layout)? {
                Some((_, align)) => {
                    // For packed types, we need to cap alignment.
                    let align = if let ty::Adt(def, _) = base.layout().ty.kind()
                        && let Some(packed) = def.repr().pack
                    {
                        align.min(packed)
                    } else {
                        align
                    };
                    (base_meta, offset.align_to(align))
                }
                None if offset == Size::ZERO => {
                    // If the offset is 0, then rounding it up to alignment wouldn't change anything,
                    // so we can do this even for types where we cannot determine the alignment.
                    (base_meta, offset)
                }
                None => {
                    // We cannot know the alignment of this field, so we cannot adjust.
                    throw_unsup!(ExternTypeField)
                }
            }
        } else {
            // base_meta could be present; we might be accessing a sized field of an unsized
            // struct.
            (MemPlaceMeta::None, offset)
        };

        base.offset_with_meta(offset, OffsetMode::Inbounds, meta, field_layout, self)
    }

    /// Downcasting to an enum variant.
    pub fn project_downcast<P: Projectable<'tcx, M::Provenance>>(
        &self,
        base: &P,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, P> {
        assert!(!base.meta().has_meta());
        // Downcasts only change the layout.
        // (In particular, no check about whether this is even the active variant -- that's by design,
        // see https://github.com/rust-lang/rust/issues/93688#issuecomment-1032929496.)
        // So we just "offset" by 0.
        let layout = base.layout().for_variant(self, variant);
        // This variant may in fact be uninhabited.
        // See <https://github.com/rust-lang/rust/issues/120337>.

        // This cannot be `transmute` as variants *can* have a smaller size than the entire enum.
        base.offset(Size::ZERO, layout, self)
    }

    /// Compute the offset and field layout for accessing the given index.
    pub fn project_index<P: Projectable<'tcx, M::Provenance>>(
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
                // With raw slices, `len` can be so big that this *can* overflow.
                let offset = self
                    .compute_size_in_bytes(stride, index)
                    .ok_or_else(|| err_ub!(PointerArithOverflow))?;

                // All fields have the same layout.
                let field_layout = base.layout().field(self, 0);
                (offset, field_layout)
            }
            _ => span_bug!(
                self.cur_span(),
                "`project_index` called on non-array type {:?}",
                base.layout().ty
            ),
        };

        base.offset(offset, field_layout, self)
    }

    /// Converts a repr(simd) value into an array of the right size, such that `project_index`
    /// accesses the SIMD elements. Also returns the number of elements.
    pub fn project_to_simd<P: Projectable<'tcx, M::Provenance>>(
        &self,
        base: &P,
    ) -> InterpResult<'tcx, (P, u64)> {
        assert!(base.layout().ty.ty_adt_def().unwrap().repr().simd());
        // SIMD types must be newtypes around arrays, so all we have to do is project to their only field.
        let array = self.project_field(base, FieldIdx::ZERO)?;
        let len = array.len(self)?;
        interp_ok((array, len))
    }

    fn project_constant_index<P: Projectable<'tcx, M::Provenance>>(
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
    /// same by repeatedly calling `project_index`.
    pub fn project_array_fields<'a, P: Projectable<'tcx, M::Provenance>>(
        &self,
        base: &'a P,
    ) -> InterpResult<'tcx, ArrayIterator<'a, 'tcx, M::Provenance, P>> {
        let abi::FieldsShape::Array { stride, .. } = base.layout().fields else {
            span_bug!(self.cur_span(), "project_array_fields: expected an array layout");
        };
        let len = base.len(self)?;
        let field_layout = base.layout().field(self, 0);
        // Ensure that all the offsets are in-bounds once, up-front.
        debug!("project_array_fields: {base:?} {len}");
        base.offset(len * stride, self.layout_of(self.tcx.types.unit).unwrap(), self)?;
        // Create the iterator.
        interp_ok(ArrayIterator {
            base,
            range: 0..len,
            stride,
            field_layout,
            _phantom: PhantomData,
        })
    }

    /// Subslicing
    fn project_subslice<P: Projectable<'tcx, M::Provenance>>(
        &self,
        base: &P,
        from: u64,
        to: u64,
        from_end: bool,
    ) -> InterpResult<'tcx, P> {
        let len = base.len(self)?; // also asserts that we have a type where this makes sense
        let actual_to = if from_end {
            if from.checked_add(to).is_none_or(|to| to > len) {
                // This can only be reached in ConstProp and non-rustc-MIR.
                throw_ub!(BoundsCheckFailed { len, index: from.saturating_add(to) });
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

        base.offset_with_meta(from_offset, OffsetMode::Inbounds, meta, layout, self)
    }

    /// Applying a general projection
    #[instrument(skip(self), level = "trace")]
    pub fn project<P>(&self, base: &P, proj_elem: mir::PlaceElem<'tcx>) -> InterpResult<'tcx, P>
    where
        P: Projectable<'tcx, M::Provenance> + From<MPlaceTy<'tcx, M::Provenance>> + std::fmt::Debug,
    {
        use rustc_middle::mir::ProjectionElem::*;
        interp_ok(match proj_elem {
            OpaqueCast(ty) => {
                span_bug!(self.cur_span(), "OpaqueCast({ty}) encountered after borrowck")
            }
            UnwrapUnsafeBinder(target) => base.transmute(self.layout_of(target)?, self)?,
            // We don't want anything happening here, this is here as a dummy.
            Subtype(_) => base.transmute(base.layout(), self)?,
            Field(field, _) => self.project_field(base, field)?,
            Downcast(_, variant) => self.project_downcast(base, variant)?,
            Deref => self.deref_pointer(&base.to_op(self)?)?.into(),
            Index(local) => {
                let layout = self.layout_of(self.tcx.types.usize)?;
                let n = self.local_to_op(local, Some(layout))?;
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
