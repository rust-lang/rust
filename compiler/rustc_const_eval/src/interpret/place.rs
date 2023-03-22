//! Computations on places -- field projections, going from mir::Place, and writing
//! into a place.
//! All high-level functions to write to memory work on places as destinations.

use either::{Either, Left, Right};

use rustc_ast::Mutability;
use rustc_middle::mir;
use rustc_middle::ty;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_target::abi::{self, Abi, Align, HasDataLayout, Size, VariantIdx};

use super::{
    alloc_range, mir_assign_valid_types, AllocId, AllocRef, AllocRefMut, CheckInAllocMsg,
    ConstAlloc, ImmTy, Immediate, InterpCx, InterpResult, Machine, MemoryKind, OpTy, Operand,
    Pointer, Provenance, Scalar,
};

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
/// Information required for the sound usage of a `MemPlace`.
pub enum MemPlaceMeta<Prov: Provenance = AllocId> {
    /// The unsized payload (e.g. length for slices or vtable pointer for trait objects).
    Meta(Scalar<Prov>),
    /// `Sized` types or unsized `extern type`
    None,
}

impl<Prov: Provenance> MemPlaceMeta<Prov> {
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn unwrap_meta(self) -> Scalar<Prov> {
        match self {
            Self::Meta(s) => s,
            Self::None => {
                bug!("expected wide pointer extra data (e.g. slice length or trait object vtable)")
            }
        }
    }

    pub fn has_meta(self) -> bool {
        match self {
            Self::Meta(_) => true,
            Self::None => false,
        }
    }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct MemPlace<Prov: Provenance = AllocId> {
    /// The pointer can be a pure integer, with the `None` provenance.
    pub ptr: Pointer<Option<Prov>>,
    /// Metadata for unsized places. Interpretation is up to the type.
    /// Must not be present for sized types, but can be missing for unsized types
    /// (e.g., `extern type`).
    pub meta: MemPlaceMeta<Prov>,
}

/// A MemPlace with its layout. Constructing it is only possible in this module.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct MPlaceTy<'tcx, Prov: Provenance = AllocId> {
    mplace: MemPlace<Prov>,
    pub layout: TyAndLayout<'tcx>,
    /// rustc does not have a proper way to represent the type of a field of a `repr(packed)` struct:
    /// it needs to have a different alignment than the field type would usually have.
    /// So we represent this here with a separate field that "overwrites" `layout.align`.
    /// This means `layout.align` should never be used for a `MPlaceTy`!
    pub align: Align,
}

#[derive(Copy, Clone, Debug)]
pub enum Place<Prov: Provenance = AllocId> {
    /// A place referring to a value allocated in the `Memory` system.
    Ptr(MemPlace<Prov>),

    /// To support alloc-free locals, we are able to write directly to a local.
    /// (Without that optimization, we'd just always be a `MemPlace`.)
    Local { frame: usize, local: mir::Local },
}

#[derive(Clone, Debug)]
pub struct PlaceTy<'tcx, Prov: Provenance = AllocId> {
    place: Place<Prov>, // Keep this private; it helps enforce invariants.
    pub layout: TyAndLayout<'tcx>,
    /// rustc does not have a proper way to represent the type of a field of a `repr(packed)` struct:
    /// it needs to have a different alignment than the field type would usually have.
    /// So we represent this here with a separate field that "overwrites" `layout.align`.
    /// This means `layout.align` should never be used for a `PlaceTy`!
    pub align: Align,
}

impl<'tcx, Prov: Provenance> std::ops::Deref for PlaceTy<'tcx, Prov> {
    type Target = Place<Prov>;
    #[inline(always)]
    fn deref(&self) -> &Place<Prov> {
        &self.place
    }
}

impl<'tcx, Prov: Provenance> std::ops::Deref for MPlaceTy<'tcx, Prov> {
    type Target = MemPlace<Prov>;
    #[inline(always)]
    fn deref(&self) -> &MemPlace<Prov> {
        &self.mplace
    }
}

impl<'tcx, Prov: Provenance> From<MPlaceTy<'tcx, Prov>> for PlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: MPlaceTy<'tcx, Prov>) -> Self {
        PlaceTy { place: Place::Ptr(*mplace), layout: mplace.layout, align: mplace.align }
    }
}

impl<'tcx, Prov: Provenance> From<&'_ MPlaceTy<'tcx, Prov>> for PlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: &MPlaceTy<'tcx, Prov>) -> Self {
        PlaceTy { place: Place::Ptr(**mplace), layout: mplace.layout, align: mplace.align }
    }
}

impl<'tcx, Prov: Provenance> From<&'_ mut MPlaceTy<'tcx, Prov>> for PlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: &mut MPlaceTy<'tcx, Prov>) -> Self {
        PlaceTy { place: Place::Ptr(**mplace), layout: mplace.layout, align: mplace.align }
    }
}

impl<Prov: Provenance> MemPlace<Prov> {
    #[inline(always)]
    pub fn from_ptr(ptr: Pointer<Option<Prov>>) -> Self {
        MemPlace { ptr, meta: MemPlaceMeta::None }
    }

    /// Adjust the provenance of the main pointer (metadata is unaffected).
    pub fn map_provenance(self, f: impl FnOnce(Option<Prov>) -> Option<Prov>) -> Self {
        MemPlace { ptr: self.ptr.map_provenance(f), ..self }
    }

    /// Turn a mplace into a (thin or wide) pointer, as a reference, pointing to the same space.
    /// This is the inverse of `ref_to_mplace`.
    #[inline(always)]
    pub fn to_ref(self, cx: &impl HasDataLayout) -> Immediate<Prov> {
        match self.meta {
            MemPlaceMeta::None => Immediate::from(Scalar::from_maybe_pointer(self.ptr, cx)),
            MemPlaceMeta::Meta(meta) => {
                Immediate::ScalarPair(Scalar::from_maybe_pointer(self.ptr, cx), meta)
            }
        }
    }

    #[inline]
    pub(super) fn offset_with_meta<'tcx>(
        self,
        offset: Size,
        meta: MemPlaceMeta<Prov>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self> {
        debug_assert!(
            !meta.has_meta() || self.meta.has_meta(),
            "cannot use `offset_with_meta` to add metadata to a place"
        );
        Ok(MemPlace { ptr: self.ptr.offset(offset, cx)?, meta })
    }
}

impl<Prov: Provenance> Place<Prov> {
    /// Asserts that this points to some local variable.
    /// Returns the frame idx and the variable idx.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_local(&self) -> (usize, mir::Local) {
        match self {
            Place::Local { frame, local } => (*frame, *local),
            _ => bug!("assert_local: expected Place::Local, got {:?}", self),
        }
    }
}

impl<'tcx, Prov: Provenance> MPlaceTy<'tcx, Prov> {
    /// Produces a MemPlace that works for ZST but nothing else.
    /// Conceptually this is a new allocation, but it doesn't actually create an allocation so you
    /// don't need to worry about memory leaks.
    #[inline]
    pub fn fake_alloc_zst(layout: TyAndLayout<'tcx>) -> Self {
        assert!(layout.is_zst());
        let align = layout.align.abi;
        let ptr = Pointer::from_addr_invalid(align.bytes()); // no provenance, absolute address
        MPlaceTy { mplace: MemPlace { ptr, meta: MemPlaceMeta::None }, layout, align }
    }

    /// Offset the place in memory and change its metadata.
    ///
    /// This can go wrong very easily if you give the wrong layout for the new place!
    #[inline]
    pub(crate) fn offset_with_meta(
        &self,
        offset: Size,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self> {
        Ok(MPlaceTy {
            mplace: self.mplace.offset_with_meta(offset, meta, cx)?,
            align: self.align.restrict_for_offset(offset),
            layout,
        })
    }

    /// Offset the place in memory.
    ///
    /// This can go wrong very easily if you give the wrong layout for the new place!
    pub fn offset(
        &self,
        offset: Size,
        layout: TyAndLayout<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self> {
        assert!(layout.is_sized());
        self.offset_with_meta(offset, MemPlaceMeta::None, layout, cx)
    }

    #[inline]
    pub fn from_aligned_ptr(ptr: Pointer<Option<Prov>>, layout: TyAndLayout<'tcx>) -> Self {
        MPlaceTy { mplace: MemPlace::from_ptr(ptr), layout, align: layout.align.abi }
    }

    #[inline]
    pub fn from_aligned_ptr_with_meta(
        ptr: Pointer<Option<Prov>>,
        layout: TyAndLayout<'tcx>,
        meta: MemPlaceMeta<Prov>,
    ) -> Self {
        let mut mplace = MemPlace::from_ptr(ptr);
        mplace.meta = meta;

        MPlaceTy { mplace, layout, align: layout.align.abi }
    }

    #[inline]
    pub(crate) fn len(&self, cx: &impl HasDataLayout) -> InterpResult<'tcx, u64> {
        if self.layout.is_unsized() {
            // We need to consult `meta` metadata
            match self.layout.ty.kind() {
                ty::Slice(..) | ty::Str => self.mplace.meta.unwrap_meta().to_target_usize(cx),
                _ => bug!("len not supported on unsized type {:?}", self.layout.ty),
            }
        } else {
            // Go through the layout. There are lots of types that support a length,
            // e.g., SIMD types. (But not all repr(simd) types even have FieldsShape::Array!)
            match self.layout.fields {
                abi::FieldsShape::Array { count, .. } => Ok(count),
                _ => bug!("len not supported on sized type {:?}", self.layout.ty),
            }
        }
    }
}

// These are defined here because they produce a place.
impl<'tcx, Prov: Provenance> OpTy<'tcx, Prov> {
    #[inline(always)]
    pub fn as_mplace_or_imm(&self) -> Either<MPlaceTy<'tcx, Prov>, ImmTy<'tcx, Prov>> {
        match **self {
            Operand::Indirect(mplace) => {
                Left(MPlaceTy { mplace, layout: self.layout, align: self.align.unwrap() })
            }
            Operand::Immediate(imm) => Right(ImmTy::from_immediate(imm, self.layout)),
        }
    }

    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_mem_place(&self) -> MPlaceTy<'tcx, Prov> {
        self.as_mplace_or_imm().left().unwrap_or_else(|| {
            bug!(
                "OpTy of type {} was immediate when it was expected to be an MPlace",
                self.layout.ty
            )
        })
    }
}

impl<'tcx, Prov: Provenance> PlaceTy<'tcx, Prov> {
    /// A place is either an mplace or some local.
    #[inline]
    pub fn as_mplace_or_local(&self) -> Either<MPlaceTy<'tcx, Prov>, (usize, mir::Local)> {
        match **self {
            Place::Ptr(mplace) => Left(MPlaceTy { mplace, layout: self.layout, align: self.align }),
            Place::Local { frame, local } => Right((frame, local)),
        }
    }

    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_mem_place(&self) -> MPlaceTy<'tcx, Prov> {
        self.as_mplace_or_local().left().unwrap_or_else(|| {
            bug!(
                "PlaceTy of type {} was a local when it was expected to be an MPlace",
                self.layout.ty
            )
        })
    }
}

// FIXME: Working around https://github.com/rust-lang/rust/issues/54385
impl<'mir, 'tcx: 'mir, Prov, M> InterpCx<'mir, 'tcx, M>
where
    Prov: Provenance + 'static,
    M: Machine<'mir, 'tcx, Provenance = Prov>,
{
    /// Take a value, which represents a (thin or wide) reference, and make it a place.
    /// Alignment is just based on the type. This is the inverse of `MemPlace::to_ref()`.
    ///
    /// Only call this if you are sure the place is "valid" (aligned and inbounds), or do not
    /// want to ever use the place for memory access!
    /// Generally prefer `deref_operand`.
    pub fn ref_to_mplace(
        &self,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let pointee_type =
            val.layout.ty.builtin_deref(true).expect("`ref_to_mplace` called on non-ptr type").ty;
        let layout = self.layout_of(pointee_type)?;
        let (ptr, meta) = match **val {
            Immediate::Scalar(ptr) => (ptr, MemPlaceMeta::None),
            Immediate::ScalarPair(ptr, meta) => (ptr, MemPlaceMeta::Meta(meta)),
            Immediate::Uninit => throw_ub!(InvalidUninitBytes(None)),
        };

        let mplace = MemPlace { ptr: ptr.to_pointer(self)?, meta };
        // When deref'ing a pointer, the *static* alignment given by the type is what matters.
        let align = layout.align.abi;
        Ok(MPlaceTy { mplace, layout, align })
    }

    /// Take an operand, representing a pointer, and dereference it to a place.
    #[instrument(skip(self), level = "debug")]
    pub fn deref_operand(
        &self,
        src: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let val = self.read_immediate(src)?;
        trace!("deref to {} on {:?}", val.layout.ty, *val);

        if val.layout.ty.is_box() {
            bug!("dereferencing {:?}", val.layout.ty);
        }

        let mplace = self.ref_to_mplace(&val)?;
        self.check_mplace(mplace)?;
        Ok(mplace)
    }

    #[inline]
    pub(super) fn get_place_alloc(
        &self,
        place: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Option<AllocRef<'_, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        assert!(place.layout.is_sized());
        assert!(!place.meta.has_meta());
        let size = place.layout.size;
        self.get_ptr_alloc(place.ptr, size, place.align)
    }

    #[inline]
    pub(super) fn get_place_alloc_mut(
        &mut self,
        place: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Option<AllocRefMut<'_, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        assert!(place.layout.is_sized());
        assert!(!place.meta.has_meta());
        let size = place.layout.size;
        self.get_ptr_alloc_mut(place.ptr, size, place.align)
    }

    /// Check if this mplace is dereferenceable and sufficiently aligned.
    pub fn check_mplace(&self, mplace: MPlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        let (size, align) = self
            .size_and_align_of_mplace(&mplace)?
            .unwrap_or((mplace.layout.size, mplace.layout.align.abi));
        assert!(mplace.align <= align, "dynamic alignment less strict than static one?");
        let align = if M::enforce_alignment(self).should_check() { align } else { Align::ONE };
        self.check_ptr_access_align(mplace.ptr, size, align, CheckInAllocMsg::DerefTest)?;
        Ok(())
    }

    /// Converts a repr(simd) place into a place where `place_index` accesses the SIMD elements.
    /// Also returns the number of elements.
    pub fn mplace_to_simd(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (MPlaceTy<'tcx, M::Provenance>, u64)> {
        // Basically we just transmute this place into an array following simd_size_and_type.
        // (Transmuting is okay since this is an in-memory place. We also double-check the size
        // stays the same.)
        let (len, e_ty) = mplace.layout.ty.simd_size_and_type(*self.tcx);
        let array = self.tcx.mk_array(e_ty, len);
        let layout = self.layout_of(array)?;
        assert_eq!(layout.size, mplace.layout.size);
        Ok((MPlaceTy { layout, ..*mplace }, len))
    }

    /// Converts a repr(simd) place into a place where `place_index` accesses the SIMD elements.
    /// Also returns the number of elements.
    pub fn place_to_simd(
        &mut self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (MPlaceTy<'tcx, M::Provenance>, u64)> {
        let mplace = self.force_allocation(place)?;
        self.mplace_to_simd(&mplace)
    }

    pub fn local_to_place(
        &self,
        frame: usize,
        local: mir::Local,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        let layout = self.layout_of_local(&self.stack()[frame], local, None)?;
        let place = Place::Local { frame, local };
        Ok(PlaceTy { place, layout, align: layout.align.abi })
    }

    /// Computes a place. You should only use this if you intend to write into this
    /// place; for reading, a more efficient alternative is `eval_place_to_op`.
    #[instrument(skip(self), level = "debug")]
    pub fn eval_place(
        &mut self,
        mir_place: mir::Place<'tcx>,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        let mut place = self.local_to_place(self.frame_idx(), mir_place.local)?;
        // Using `try_fold` turned out to be bad for performance, hence the loop.
        for elem in mir_place.projection.iter() {
            place = self.place_projection(&place, elem)?
        }

        trace!("{:?}", self.dump_place(place.place));
        // Sanity-check the type we ended up with.
        debug_assert!(
            mir_assign_valid_types(
                *self.tcx,
                self.param_env,
                self.layout_of(self.subst_from_current_frame_and_normalize_erasing_regions(
                    mir_place.ty(&self.frame().body.local_decls, *self.tcx).ty
                )?)?,
                place.layout,
            ),
            "eval_place of a MIR place with type {:?} produced an interpreter place with type {:?}",
            mir_place.ty(&self.frame().body.local_decls, *self.tcx).ty,
            place.layout.ty,
        );
        Ok(place)
    }

    /// Write an immediate to a place
    #[inline(always)]
    #[instrument(skip(self), level = "debug")]
    pub fn write_immediate(
        &mut self,
        src: Immediate<M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_immediate_no_validate(src, dest)?;

        if M::enforce_validity(self, dest.layout) {
            // Data got changed, better make sure it matches the type!
            self.validate_operand(&self.place_to_op(dest)?)?;
        }

        Ok(())
    }

    /// Write a scalar to a place
    #[inline(always)]
    pub fn write_scalar(
        &mut self,
        val: impl Into<Scalar<M::Provenance>>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_immediate(Immediate::Scalar(val.into()), dest)
    }

    /// Write a pointer to a place
    #[inline(always)]
    pub fn write_pointer(
        &mut self,
        ptr: impl Into<Pointer<Option<M::Provenance>>>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_scalar(Scalar::from_maybe_pointer(ptr.into(), self), dest)
    }

    /// Write an immediate to a place.
    /// If you use this you are responsible for validating that things got copied at the
    /// right type.
    fn write_immediate_no_validate(
        &mut self,
        src: Immediate<M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        assert!(dest.layout.is_sized(), "Cannot write unsized data");
        trace!("write_immediate: {:?} <- {:?}: {}", *dest, src, dest.layout.ty);

        // See if we can avoid an allocation. This is the counterpart to `read_immediate_raw`,
        // but not factored as a separate function.
        let mplace = match dest.place {
            Place::Local { frame, local } => {
                match M::access_local_mut(self, frame, local)? {
                    Operand::Immediate(local) => {
                        // Local can be updated in-place.
                        *local = src;
                        return Ok(());
                    }
                    Operand::Indirect(mplace) => {
                        // The local is in memory, go on below.
                        *mplace
                    }
                }
            }
            Place::Ptr(mplace) => mplace, // already referring to memory
        };

        // This is already in memory, write there.
        self.write_immediate_to_mplace_no_validate(src, dest.layout, dest.align, mplace)
    }

    /// Write an immediate to memory.
    /// If you use this you are responsible for validating that things got copied at the
    /// right layout.
    fn write_immediate_to_mplace_no_validate(
        &mut self,
        value: Immediate<M::Provenance>,
        layout: TyAndLayout<'tcx>,
        align: Align,
        dest: MemPlace<M::Provenance>,
    ) -> InterpResult<'tcx> {
        // Note that it is really important that the type here is the right one, and matches the
        // type things are read at. In case `value` is a `ScalarPair`, we don't do any magic here
        // to handle padding properly, which is only correct if we never look at this data with the
        // wrong type.

        let tcx = *self.tcx;
        let Some(mut alloc) = self.get_place_alloc_mut(&MPlaceTy { mplace: dest, layout, align })? else {
            // zero-sized access
            return Ok(());
        };

        match value {
            Immediate::Scalar(scalar) => {
                let Abi::Scalar(s) = layout.abi else { span_bug!(
                        self.cur_span(),
                        "write_immediate_to_mplace: invalid Scalar layout: {layout:#?}",
                    )
                };
                let size = s.size(&tcx);
                assert_eq!(size, layout.size, "abi::Scalar size does not match layout size");
                alloc.write_scalar(alloc_range(Size::ZERO, size), scalar)
            }
            Immediate::ScalarPair(a_val, b_val) => {
                // We checked `ptr_align` above, so all fields will have the alignment they need.
                // We would anyway check against `ptr_align.restrict_for_offset(b_offset)`,
                // which `ptr.offset(b_offset)` cannot possibly fail to satisfy.
                let Abi::ScalarPair(a, b) = layout.abi else { span_bug!(
                        self.cur_span(),
                        "write_immediate_to_mplace: invalid ScalarPair layout: {:#?}",
                        layout
                    )
                };
                let (a_size, b_size) = (a.size(&tcx), b.size(&tcx));
                let b_offset = a_size.align_to(b.align(&tcx).abi);
                assert!(b_offset.bytes() > 0); // in `operand_field` we use the offset to tell apart the fields

                // It is tempting to verify `b_offset` against `layout.fields.offset(1)`,
                // but that does not work: We could be a newtype around a pair, then the
                // fields do not match the `ScalarPair` components.

                alloc.write_scalar(alloc_range(Size::ZERO, a_size), a_val)?;
                alloc.write_scalar(alloc_range(b_offset, b_size), b_val)
            }
            Immediate::Uninit => alloc.write_uninit(),
        }
    }

    pub fn write_uninit(&mut self, dest: &PlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        let mplace = match dest.as_mplace_or_local() {
            Left(mplace) => mplace,
            Right((frame, local)) => {
                match M::access_local_mut(self, frame, local)? {
                    Operand::Immediate(local) => {
                        *local = Immediate::Uninit;
                        return Ok(());
                    }
                    Operand::Indirect(mplace) => {
                        // The local is in memory, go on below.
                        MPlaceTy { mplace: *mplace, layout: dest.layout, align: dest.align }
                    }
                }
            }
        };
        let Some(mut alloc) = self.get_place_alloc_mut(&mplace)? else {
            // Zero-sized access
            return Ok(());
        };
        alloc.write_uninit()?;
        Ok(())
    }

    /// Copies the data from an operand to a place.
    /// `allow_transmute` indicates whether the layouts may disagree.
    #[inline(always)]
    #[instrument(skip(self), level = "debug")]
    pub fn copy_op(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
        allow_transmute: bool,
    ) -> InterpResult<'tcx> {
        self.copy_op_no_validate(src, dest, allow_transmute)?;

        if M::enforce_validity(self, dest.layout) {
            // Data got changed, better make sure it matches the type!
            self.validate_operand(&self.place_to_op(dest)?)?;
        }

        Ok(())
    }

    /// Copies the data from an operand to a place.
    /// `allow_transmute` indicates whether the layouts may disagree.
    /// Also, if you use this you are responsible for validating that things get copied at the
    /// right type.
    #[instrument(skip(self), level = "debug")]
    fn copy_op_no_validate(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
        allow_transmute: bool,
    ) -> InterpResult<'tcx> {
        // We do NOT compare the types for equality, because well-typed code can
        // actually "transmute" `&mut T` to `&T` in an assignment without a cast.
        let layout_compat =
            mir_assign_valid_types(*self.tcx, self.param_env, src.layout, dest.layout);
        if !allow_transmute && !layout_compat {
            span_bug!(
                self.cur_span(),
                "type mismatch when copying!\nsrc: {:?},\ndest: {:?}",
                src.layout.ty,
                dest.layout.ty,
            );
        }

        // Let us see if the layout is simple so we take a shortcut,
        // avoid force_allocation.
        let src = match self.read_immediate_raw(src)? {
            Right(src_val) => {
                // FIXME(const_prop): Const-prop can possibly evaluate an
                // unsized copy operation when it thinks that the type is
                // actually sized, due to a trivially false where-clause
                // predicate like `where Self: Sized` with `Self = dyn Trait`.
                // See #102553 for an example of such a predicate.
                if src.layout.is_unsized() {
                    throw_inval!(SizeOfUnsizedType(src.layout.ty));
                }
                if dest.layout.is_unsized() {
                    throw_inval!(SizeOfUnsizedType(dest.layout.ty));
                }
                assert_eq!(src.layout.size, dest.layout.size);
                // Yay, we got a value that we can write directly.
                return if layout_compat {
                    self.write_immediate_no_validate(*src_val, dest)
                } else {
                    // This is tricky. The problematic case is `ScalarPair`: the `src_val` was
                    // loaded using the offsets defined by `src.layout`. When we put this back into
                    // the destination, we have to use the same offsets! So (a) we make sure we
                    // write back to memory, and (b) we use `dest` *with the source layout*.
                    let dest_mem = self.force_allocation(dest)?;
                    self.write_immediate_to_mplace_no_validate(
                        *src_val,
                        src.layout,
                        dest_mem.align,
                        *dest_mem,
                    )
                };
            }
            Left(mplace) => mplace,
        };
        // Slow path, this does not fit into an immediate. Just memcpy.
        trace!("copy_op: {:?} <- {:?}: {}", *dest, src, dest.layout.ty);

        let dest = self.force_allocation(&dest)?;
        let Some((dest_size, _)) = self.size_and_align_of_mplace(&dest)? else {
            span_bug!(self.cur_span(), "copy_op needs (dynamically) sized values")
        };
        if cfg!(debug_assertions) {
            let src_size = self.size_and_align_of_mplace(&src)?.unwrap().0;
            assert_eq!(src_size, dest_size, "Cannot copy differently-sized data");
        } else {
            // As a cheap approximation, we compare the fixed parts of the size.
            assert_eq!(src.layout.size, dest.layout.size);
        }

        self.mem_copy(
            src.ptr, src.align, dest.ptr, dest.align, dest_size, /*nonoverlapping*/ false,
        )
    }

    /// Ensures that a place is in memory, and returns where it is.
    /// If the place currently refers to a local that doesn't yet have a matching allocation,
    /// create such an allocation.
    /// This is essentially `force_to_memplace`.
    #[instrument(skip(self), level = "debug")]
    pub fn force_allocation(
        &mut self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let mplace = match place.place {
            Place::Local { frame, local } => {
                match M::access_local_mut(self, frame, local)? {
                    &mut Operand::Immediate(local_val) => {
                        // We need to make an allocation.

                        // We need the layout of the local. We can NOT use the layout we got,
                        // that might e.g., be an inner field of a struct with `Scalar` layout,
                        // that has different alignment than the outer field.
                        let local_layout =
                            self.layout_of_local(&self.stack()[frame], local, None)?;
                        if local_layout.is_unsized() {
                            throw_unsup_format!("unsized locals are not supported");
                        }
                        let mplace = *self.allocate(local_layout, MemoryKind::Stack)?;
                        if !matches!(local_val, Immediate::Uninit) {
                            // Preserve old value. (As an optimization, we can skip this if it was uninit.)
                            // We don't have to validate as we can assume the local
                            // was already valid for its type.
                            self.write_immediate_to_mplace_no_validate(
                                local_val,
                                local_layout,
                                local_layout.align.abi,
                                mplace,
                            )?;
                        }
                        // Now we can call `access_mut` again, asserting it goes well,
                        // and actually overwrite things.
                        *M::access_local_mut(self, frame, local).unwrap() =
                            Operand::Indirect(mplace);
                        mplace
                    }
                    &mut Operand::Indirect(mplace) => mplace, // this already was an indirect local
                }
            }
            Place::Ptr(mplace) => mplace,
        };
        // Return with the original layout, so that the caller can go on
        Ok(MPlaceTy { mplace, layout: place.layout, align: place.align })
    }

    pub fn allocate(
        &mut self,
        layout: TyAndLayout<'tcx>,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        assert!(layout.is_sized());
        let ptr = self.allocate_ptr(layout.size, layout.align.abi, kind)?;
        Ok(MPlaceTy::from_aligned_ptr(ptr.into(), layout))
    }

    /// Returns a wide MPlace of type `&'static [mut] str` to a new 1-aligned allocation.
    pub fn allocate_str(
        &mut self,
        str: &str,
        kind: MemoryKind<M::MemoryKind>,
        mutbl: Mutability,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let ptr = self.allocate_bytes_ptr(str.as_bytes(), Align::ONE, kind, mutbl)?;
        let meta = Scalar::from_target_usize(u64::try_from(str.len()).unwrap(), self);
        let mplace = MemPlace { ptr: ptr.into(), meta: MemPlaceMeta::Meta(meta) };

        let ty = self.tcx.mk_ref(
            self.tcx.lifetimes.re_static,
            ty::TypeAndMut { ty: self.tcx.types.str_, mutbl },
        );
        let layout = self.layout_of(ty).unwrap();
        Ok(MPlaceTy { mplace, layout, align: layout.align.abi })
    }

    /// Writes the aggregate to the destination.
    #[instrument(skip(self), level = "trace")]
    pub fn write_aggregate(
        &mut self,
        kind: &mir::AggregateKind<'tcx>,
        operands: &[mir::Operand<'tcx>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_uninit(&dest)?;
        let (variant_index, variant_dest, active_field_index) = match *kind {
            mir::AggregateKind::Adt(_, variant_index, _, _, active_field_index) => {
                let variant_dest = self.place_downcast(&dest, variant_index)?;
                (variant_index, variant_dest, active_field_index)
            }
            _ => (VariantIdx::from_u32(0), dest.clone(), None),
        };
        if active_field_index.is_some() {
            assert_eq!(operands.len(), 1);
        }
        for (field_index, operand) in operands.iter().enumerate() {
            let field_index = active_field_index.unwrap_or(field_index);
            let field_dest = self.place_field(&variant_dest, field_index)?;
            let op = self.eval_operand(operand, Some(field_dest.layout))?;
            self.copy_op(&op, &field_dest, /*allow_transmute*/ false)?;
        }
        self.write_discriminant(variant_index, &dest)
    }

    pub fn raw_const_to_mplace(
        &self,
        raw: ConstAlloc<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        // This must be an allocation in `tcx`
        let _ = self.tcx.global_alloc(raw.alloc_id);
        let ptr = self.global_base_pointer(Pointer::from(raw.alloc_id))?;
        let layout = self.layout_of(raw.ty)?;
        Ok(MPlaceTy::from_aligned_ptr(ptr.into(), layout))
    }

    /// Turn a place with a `dyn Trait` type into a place with the actual dynamic type.
    /// Aso returns the vtable.
    pub(super) fn unpack_dyn_trait(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (MPlaceTy<'tcx, M::Provenance>, Pointer<Option<M::Provenance>>)> {
        assert!(
            matches!(mplace.layout.ty.kind(), ty::Dynamic(_, _, ty::Dyn)),
            "`unpack_dyn_trait` only makes sense on `dyn*` types"
        );
        let vtable = mplace.meta.unwrap_meta().to_pointer(self)?;
        let (ty, _) = self.get_ptr_vtable(vtable)?;
        let layout = self.layout_of(ty)?;

        let mplace = MPlaceTy {
            mplace: MemPlace { meta: MemPlaceMeta::None, ..**mplace },
            layout,
            align: layout.align.abi,
        };
        Ok((mplace, vtable))
    }

    /// Turn an operand with a `dyn* Trait` type into an operand with the actual dynamic type.
    /// Aso returns the vtable.
    pub(super) fn unpack_dyn_star(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (OpTy<'tcx, M::Provenance>, Pointer<Option<M::Provenance>>)> {
        assert!(
            matches!(op.layout.ty.kind(), ty::Dynamic(_, _, ty::DynStar)),
            "`unpack_dyn_star` only makes sense on `dyn*` types"
        );
        let data = self.operand_field(&op, 0)?;
        let vtable = self.operand_field(&op, 1)?;
        let vtable = self.read_pointer(&vtable)?;
        let (ty, _) = self.get_ptr_vtable(vtable)?;
        let layout = self.layout_of(ty)?;
        let data = data.transmute(layout);
        Ok((data, vtable))
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(MemPlace, 40);
    static_assert_size!(MemPlaceMeta, 24);
    static_assert_size!(MPlaceTy<'_>, 64);
    static_assert_size!(Place, 40);
    static_assert_size!(PlaceTy<'_>, 64);
    // tidy-alphabetical-end
}
