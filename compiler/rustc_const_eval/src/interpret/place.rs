//! Computations on places -- field projections, going from mir::Place, and writing
//! into a place.
//! All high-level functions to write to memory work on places as destinations.

use std::assert_matches::assert_matches;

use either::{Either, Left, Right};
use rustc_abi::{BackendRepr, HasDataLayout, Size};
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::{bug, mir, span_bug};
use tracing::field::Empty;
use tracing::{instrument, trace};

use super::{
    AllocInit, AllocRef, AllocRefMut, CheckAlignMsg, CtfeProvenance, ImmTy, Immediate, InterpCx,
    InterpResult, Machine, MemoryKind, Misalignment, OffsetMode, OpTy, Operand, Pointer,
    Projectable, Provenance, Scalar, alloc_range, interp_ok, mir_assign_valid_types,
};
use crate::enter_trace_span;

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
/// Information required for the sound usage of a `MemPlace`.
pub enum MemPlaceMeta<Prov: Provenance = CtfeProvenance> {
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

    #[inline(always)]
    pub fn has_meta(self) -> bool {
        match self {
            Self::Meta(_) => true,
            Self::None => false,
        }
    }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub(super) struct MemPlace<Prov: Provenance = CtfeProvenance> {
    /// The pointer can be a pure integer, with the `None` provenance.
    pub ptr: Pointer<Option<Prov>>,
    /// Metadata for unsized places. Interpretation is up to the type.
    /// Must not be present for sized types, but can be missing for unsized types
    /// (e.g., `extern type`).
    pub meta: MemPlaceMeta<Prov>,
    /// Stores whether this place was created based on a sufficiently aligned pointer.
    misaligned: Option<Misalignment>,
}

impl<Prov: Provenance> MemPlace<Prov> {
    /// Adjust the provenance of the main pointer (metadata is unaffected).
    fn map_provenance(self, f: impl FnOnce(Prov) -> Prov) -> Self {
        MemPlace { ptr: self.ptr.map_provenance(|p| p.map(f)), ..self }
    }

    /// Turn a mplace into a (thin or wide) pointer, as a reference, pointing to the same space.
    #[inline]
    fn to_ref(self, cx: &impl HasDataLayout) -> Immediate<Prov> {
        Immediate::new_pointer_with_meta(self.ptr, self.meta, cx)
    }

    #[inline]
    // Not called `offset_with_meta` to avoid confusion with the trait method.
    fn offset_with_meta_<'tcx, M: Machine<'tcx, Provenance = Prov>>(
        self,
        offset: Size,
        mode: OffsetMode,
        meta: MemPlaceMeta<Prov>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self> {
        debug_assert!(
            !meta.has_meta() || self.meta.has_meta(),
            "cannot use `offset_with_meta` to add metadata to a place"
        );
        let ptr = match mode {
            OffsetMode::Inbounds => {
                ecx.ptr_offset_inbounds(self.ptr, offset.bytes().try_into().unwrap())?
            }
            OffsetMode::Wrapping => self.ptr.wrapping_offset(offset, ecx),
        };
        interp_ok(MemPlace { ptr, meta, misaligned: self.misaligned })
    }
}

/// A MemPlace with its layout. Constructing it is only possible in this module.
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct MPlaceTy<'tcx, Prov: Provenance = CtfeProvenance> {
    mplace: MemPlace<Prov>,
    pub layout: TyAndLayout<'tcx>,
}

impl<Prov: Provenance> std::fmt::Debug for MPlaceTy<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Printing `layout` results in too much noise; just print a nice version of the type.
        f.debug_struct("MPlaceTy")
            .field("mplace", &self.mplace)
            .field("ty", &format_args!("{}", self.layout.ty))
            .finish()
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
        let ptr = Pointer::without_provenance(align.bytes()); // no provenance, absolute address
        MPlaceTy { mplace: MemPlace { ptr, meta: MemPlaceMeta::None, misaligned: None }, layout }
    }

    /// Adjust the provenance of the main pointer (metadata is unaffected).
    pub fn map_provenance(self, f: impl FnOnce(Prov) -> Prov) -> Self {
        MPlaceTy { mplace: self.mplace.map_provenance(f), ..self }
    }

    #[inline(always)]
    pub(super) fn mplace(&self) -> &MemPlace<Prov> {
        &self.mplace
    }

    #[inline(always)]
    pub fn ptr(&self) -> Pointer<Option<Prov>> {
        self.mplace.ptr
    }

    #[inline(always)]
    pub fn to_ref(&self, cx: &impl HasDataLayout) -> Immediate<Prov> {
        self.mplace.to_ref(cx)
    }
}

impl<'tcx, Prov: Provenance> Projectable<'tcx, Prov> for MPlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn meta(&self) -> MemPlaceMeta<Prov> {
        self.mplace.meta
    }

    fn offset_with_meta<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        offset: Size,
        mode: OffsetMode,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self> {
        interp_ok(MPlaceTy {
            mplace: self.mplace.offset_with_meta_(offset, mode, meta, ecx)?,
            layout,
        })
    }

    #[inline(always)]
    fn to_op<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        _ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        interp_ok(self.clone().into())
    }
}

#[derive(Copy, Clone, Debug)]
pub(super) enum Place<Prov: Provenance = CtfeProvenance> {
    /// A place referring to a value allocated in the `Memory` system.
    Ptr(MemPlace<Prov>),

    /// To support alloc-free locals, we are able to write directly to a local. The offset indicates
    /// where in the local this place is located; if it is `None`, no projection has been applied
    /// and the type of the place is exactly the type of the local.
    /// Such projections are meaningful even if the offset is 0, since they can change layouts.
    /// (Without that optimization, we'd just always be a `MemPlace`.)
    /// `Local` places always refer to the current stack frame, so they are unstable under
    /// function calls/returns and switching betweens stacks of different threads!
    /// We carry around the address of the `locals` buffer of the correct stack frame as a sanity
    /// check to be able to catch some cases of using a dangling `Place`.
    ///
    /// This variant shall not be used for unsized types -- those must always live in memory.
    Local { local: mir::Local, offset: Option<Size>, locals_addr: usize },
}

/// An evaluated place, together with its type.
///
/// This may reference a stack frame by its index, so `PlaceTy` should generally not be kept around
/// for longer than a single operation. Popping and then pushing a stack frame can make `PlaceTy`
/// point to the wrong destination. If the interpreter has multiple stacks, stack switching will
/// also invalidate a `PlaceTy`.
#[derive(Clone)]
pub struct PlaceTy<'tcx, Prov: Provenance = CtfeProvenance> {
    place: Place<Prov>, // Keep this private; it helps enforce invariants.
    pub layout: TyAndLayout<'tcx>,
}

impl<Prov: Provenance> std::fmt::Debug for PlaceTy<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Printing `layout` results in too much noise; just print a nice version of the type.
        f.debug_struct("PlaceTy")
            .field("place", &self.place)
            .field("ty", &format_args!("{}", self.layout.ty))
            .finish()
    }
}

impl<'tcx, Prov: Provenance> From<MPlaceTy<'tcx, Prov>> for PlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: MPlaceTy<'tcx, Prov>) -> Self {
        PlaceTy { place: Place::Ptr(mplace.mplace), layout: mplace.layout }
    }
}

impl<'tcx, Prov: Provenance> PlaceTy<'tcx, Prov> {
    #[inline(always)]
    pub(super) fn place(&self) -> &Place<Prov> {
        &self.place
    }

    /// A place is either an mplace or some local.
    ///
    /// Note that the return value can be different even for logically identical places!
    /// Specifically, if a local is stored in-memory, this may return `Local` or `MPlaceTy`
    /// depending on how the place was constructed. In other words, seeing `Local` here does *not*
    /// imply that this place does not point to memory. Every caller must therefore always handle
    /// both cases.
    #[inline(always)]
    pub fn as_mplace_or_local(
        &self,
    ) -> Either<MPlaceTy<'tcx, Prov>, (mir::Local, Option<Size>, usize, TyAndLayout<'tcx>)> {
        match self.place {
            Place::Ptr(mplace) => Left(MPlaceTy { mplace, layout: self.layout }),
            Place::Local { local, offset, locals_addr } => {
                Right((local, offset, locals_addr, self.layout))
            }
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

impl<'tcx, Prov: Provenance> Projectable<'tcx, Prov> for PlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline]
    fn meta(&self) -> MemPlaceMeta<Prov> {
        match self.as_mplace_or_local() {
            Left(mplace) => mplace.meta(),
            Right(_) => {
                debug_assert!(self.layout.is_sized(), "unsized locals should live in memory");
                MemPlaceMeta::None
            }
        }
    }

    fn offset_with_meta<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        offset: Size,
        mode: OffsetMode,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self> {
        interp_ok(match self.as_mplace_or_local() {
            Left(mplace) => mplace.offset_with_meta(offset, mode, meta, layout, ecx)?.into(),
            Right((local, old_offset, locals_addr, _)) => {
                debug_assert!(layout.is_sized(), "unsized locals should live in memory");
                assert_matches!(meta, MemPlaceMeta::None); // we couldn't store it anyway...
                // `Place::Local` are always in-bounds of their surrounding local, so we can just
                // check directly if this remains in-bounds. This cannot actually be violated since
                // projections are type-checked and bounds-checked.
                assert!(offset + layout.size <= self.layout.size);

                // Size `+`, ensures no overflow.
                let new_offset = old_offset.unwrap_or(Size::ZERO) + offset;

                PlaceTy {
                    place: Place::Local { local, offset: Some(new_offset), locals_addr },
                    layout,
                }
            }
        })
    }

    #[inline(always)]
    fn to_op<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        ecx.place_to_op(self)
    }
}

// These are defined here because they produce a place.
impl<'tcx, Prov: Provenance> OpTy<'tcx, Prov> {
    #[inline(always)]
    pub fn as_mplace_or_imm(&self) -> Either<MPlaceTy<'tcx, Prov>, ImmTy<'tcx, Prov>> {
        match self.op() {
            Operand::Indirect(mplace) => Left(MPlaceTy { mplace: *mplace, layout: self.layout }),
            Operand::Immediate(imm) => Right(ImmTy::from_immediate(*imm, self.layout)),
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

/// The `Weiteable` trait describes interpreter values that can be written to.
pub trait Writeable<'tcx, Prov: Provenance>: Projectable<'tcx, Prov> {
    fn to_place(&self) -> PlaceTy<'tcx, Prov>;

    fn force_mplace<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        ecx: &mut InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, Prov>>;
}

impl<'tcx, Prov: Provenance> Writeable<'tcx, Prov> for PlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn to_place(&self) -> PlaceTy<'tcx, Prov> {
        self.clone()
    }

    #[inline(always)]
    fn force_mplace<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        ecx: &mut InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, Prov>> {
        ecx.force_allocation(self)
    }
}

impl<'tcx, Prov: Provenance> Writeable<'tcx, Prov> for MPlaceTy<'tcx, Prov> {
    #[inline(always)]
    fn to_place(&self) -> PlaceTy<'tcx, Prov> {
        self.clone().into()
    }

    #[inline(always)]
    fn force_mplace<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        _ecx: &mut InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, Prov>> {
        interp_ok(self.clone())
    }
}

// FIXME: Working around https://github.com/rust-lang/rust/issues/54385
impl<'tcx, Prov, M> InterpCx<'tcx, M>
where
    Prov: Provenance,
    M: Machine<'tcx, Provenance = Prov>,
{
    fn ptr_with_meta_to_mplace(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        meta: MemPlaceMeta<M::Provenance>,
        layout: TyAndLayout<'tcx>,
        unaligned: bool,
    ) -> MPlaceTy<'tcx, M::Provenance> {
        let misaligned =
            if unaligned { None } else { self.is_ptr_misaligned(ptr, layout.align.abi) };
        MPlaceTy { mplace: MemPlace { ptr, meta, misaligned }, layout }
    }

    pub fn ptr_to_mplace(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        layout: TyAndLayout<'tcx>,
    ) -> MPlaceTy<'tcx, M::Provenance> {
        assert!(layout.is_sized());
        self.ptr_with_meta_to_mplace(ptr, MemPlaceMeta::None, layout, /*unaligned*/ false)
    }

    pub fn ptr_to_mplace_unaligned(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        layout: TyAndLayout<'tcx>,
    ) -> MPlaceTy<'tcx, M::Provenance> {
        assert!(layout.is_sized());
        self.ptr_with_meta_to_mplace(ptr, MemPlaceMeta::None, layout, /*unaligned*/ true)
    }

    /// Take a value, which represents a (thin or wide) reference, and make it a place.
    /// Alignment is just based on the type. This is the inverse of `mplace_to_ref()`.
    ///
    /// Only call this if you are sure the place is "valid" (aligned and inbounds), or do not
    /// want to ever use the place for memory access!
    /// Generally prefer `deref_pointer`.
    pub fn ref_to_mplace(
        &self,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let pointee_type =
            val.layout.ty.builtin_deref(true).expect("`ref_to_mplace` called on non-ptr type");
        let layout = self.layout_of(pointee_type)?;
        let (ptr, meta) = val.to_scalar_and_meta();

        // `ref_to_mplace` is called on raw pointers even if they don't actually get dereferenced;
        // we hence can't call `size_and_align_of` since that asserts more validity than we want.
        let ptr = ptr.to_pointer(self)?;
        interp_ok(self.ptr_with_meta_to_mplace(ptr, meta, layout, /*unaligned*/ false))
    }

    /// Turn a mplace into a (thin or wide) mutable raw pointer, pointing to the same space.
    /// `align` information is lost!
    /// This is the inverse of `ref_to_mplace`.
    pub fn mplace_to_ref(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        let imm = mplace.mplace.to_ref(self);
        let layout = self.layout_of(Ty::new_mut_ptr(self.tcx.tcx, mplace.layout.ty))?;
        interp_ok(ImmTy::from_immediate(imm, layout))
    }

    /// Take an operand, representing a pointer, and dereference it to a place.
    /// Corresponds to the `*` operator in Rust.
    #[instrument(skip(self), level = "trace")]
    pub fn deref_pointer(
        &self,
        src: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        if src.layout().ty.is_box() {
            // Derefer should have removed all Box derefs.
            // Some `Box` are not immediates (if they have a custom allocator)
            // so the code below would fail.
            bug!("dereferencing {}", src.layout().ty);
        }

        let val = self.read_immediate(src)?;
        trace!("deref to {} on {:?}", val.layout.ty, *val);

        let mplace = self.ref_to_mplace(&val)?;
        interp_ok(mplace)
    }

    #[inline]
    pub(super) fn get_place_alloc(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Option<AllocRef<'_, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        let (size, _align) = self
            .size_and_align_of_val(mplace)?
            .unwrap_or((mplace.layout.size, mplace.layout.align.abi));
        // We check alignment separately, and *after* checking everything else.
        // If an access is both OOB and misaligned, we want to see the bounds error.
        let a = self.get_ptr_alloc(mplace.ptr(), size)?;
        self.check_misalign(mplace.mplace.misaligned, CheckAlignMsg::BasedOn)?;
        interp_ok(a)
    }

    #[inline]
    pub(super) fn get_place_alloc_mut(
        &mut self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Option<AllocRefMut<'_, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        let (size, _align) = self
            .size_and_align_of_val(mplace)?
            .unwrap_or((mplace.layout.size, mplace.layout.align.abi));
        // We check alignment separately, and raise that error *after* checking everything else.
        // If an access is both OOB and misaligned, we want to see the bounds error.
        // However we have to call `check_misalign` first to make the borrow checker happy.
        let misalign_res = self.check_misalign(mplace.mplace.misaligned, CheckAlignMsg::BasedOn);
        // An error from get_ptr_alloc_mut takes precedence.
        let (a, ()) = self.get_ptr_alloc_mut(mplace.ptr(), size).and(misalign_res)?;
        interp_ok(a)
    }

    /// Turn a local in the current frame into a place.
    pub fn local_to_place(
        &self,
        local: mir::Local,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        let frame = self.frame();
        let layout = self.layout_of_local(frame, local, None)?;
        let place = if layout.is_sized() {
            // We can just always use the `Local` for sized values.
            Place::Local { local, offset: None, locals_addr: frame.locals_addr() }
        } else {
            // Other parts of the system rely on `Place::Local` never being unsized.
            match frame.locals[local].access()? {
                Operand::Immediate(_) => bug!(),
                Operand::Indirect(mplace) => Place::Ptr(*mplace),
            }
        };
        interp_ok(PlaceTy { place, layout })
    }

    /// Computes a place. You should only use this if you intend to write into this
    /// place; for reading, a more efficient alternative is `eval_place_to_op`.
    #[instrument(skip(self), level = "trace")]
    pub fn eval_place(
        &self,
        mir_place: mir::Place<'tcx>,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, M::Provenance>> {
        let _trace =
            enter_trace_span!(M, step::eval_place, ?mir_place, tracing_separate_thread = Empty);

        let mut place = self.local_to_place(mir_place.local)?;
        // Using `try_fold` turned out to be bad for performance, hence the loop.
        for elem in mir_place.projection.iter() {
            place = self.project(&place, elem)?
        }

        trace!("{:?}", self.dump_place(&place));
        // Sanity-check the type we ended up with.
        if cfg!(debug_assertions) {
            let normalized_place_ty = self
                .instantiate_from_current_frame_and_normalize_erasing_regions(
                    mir_place.ty(&self.frame().body.local_decls, *self.tcx).ty,
                )?;
            if !mir_assign_valid_types(
                *self.tcx,
                self.typing_env,
                self.layout_of(normalized_place_ty)?,
                place.layout,
            ) {
                span_bug!(
                    self.cur_span(),
                    "eval_place of a MIR place with type {} produced an interpreter place with type {}",
                    normalized_place_ty,
                    place.layout.ty,
                )
            }
        }
        interp_ok(place)
    }

    /// Given a place, returns either the underlying mplace or a reference to where the value of
    /// this place is stored.
    #[inline(always)]
    fn as_mplace_or_mutable_local(
        &mut self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<
        'tcx,
        Either<
            MPlaceTy<'tcx, M::Provenance>,
            (&mut Immediate<M::Provenance>, TyAndLayout<'tcx>, mir::Local),
        >,
    > {
        interp_ok(match place.to_place().as_mplace_or_local() {
            Left(mplace) => Left(mplace),
            Right((local, offset, locals_addr, layout)) => {
                if offset.is_some() {
                    // This has been projected to a part of this local, or had the type changed.
                    // FIXME: there are cases where we could still avoid allocating an mplace.
                    Left(place.force_mplace(self)?)
                } else {
                    debug_assert_eq!(locals_addr, self.frame().locals_addr());
                    debug_assert_eq!(self.layout_of_local(self.frame(), local, None)?, layout);
                    match self.frame_mut().locals[local].access_mut()? {
                        Operand::Indirect(mplace) => {
                            // The local is in memory.
                            Left(MPlaceTy { mplace: *mplace, layout })
                        }
                        Operand::Immediate(local_val) => {
                            // The local still has the optimized representation.
                            Right((local_val, layout, local))
                        }
                    }
                }
            }
        })
    }

    /// Write an immediate to a place
    #[inline(always)]
    #[instrument(skip(self), level = "trace")]
    pub fn write_immediate(
        &mut self,
        src: Immediate<M::Provenance>,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_immediate_no_validate(src, dest)?;

        if M::enforce_validity(self, dest.layout()) {
            // Data got changed, better make sure it matches the type!
            // Also needed to reset padding.
            self.validate_operand(
                &dest.to_place(),
                M::enforce_validity_recursively(self, dest.layout()),
                /*reset_provenance_and_padding*/ true,
            )?;
        }

        interp_ok(())
    }

    /// Write a scalar to a place
    #[inline(always)]
    pub fn write_scalar(
        &mut self,
        val: impl Into<Scalar<M::Provenance>>,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_immediate(Immediate::Scalar(val.into()), dest)
    }

    /// Write a pointer to a place
    #[inline(always)]
    pub fn write_pointer(
        &mut self,
        ptr: impl Into<Pointer<Option<M::Provenance>>>,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_scalar(Scalar::from_maybe_pointer(ptr.into(), self), dest)
    }

    /// Write an immediate to a place.
    /// If you use this you are responsible for validating that things got copied at the
    /// right type.
    pub(super) fn write_immediate_no_validate(
        &mut self,
        src: Immediate<M::Provenance>,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        assert!(dest.layout().is_sized(), "Cannot write unsized immediate data");

        match self.as_mplace_or_mutable_local(&dest.to_place())? {
            Right((local_val, local_layout, local)) => {
                // Local can be updated in-place.
                *local_val = src;
                // Call the machine hook (the data race detector needs to know about this write).
                if !self.validation_in_progress() {
                    M::after_local_write(self, local, /*storage_live*/ false)?;
                }
                // Double-check that the value we are storing and the local fit to each other.
                // Things can ge wrong in quite weird ways when this is violated.
                // Unfortunately this is too expensive to do in release builds.
                if cfg!(debug_assertions) {
                    src.assert_matches_abi(
                        local_layout.backend_repr,
                        "invalid immediate for given destination place",
                        self,
                    );
                }
            }
            Left(mplace) => {
                self.write_immediate_to_mplace_no_validate(src, mplace.layout, mplace.mplace)?;
            }
        }
        interp_ok(())
    }

    /// Write an immediate to memory.
    /// If you use this you are responsible for validating that things got copied at the
    /// right layout.
    fn write_immediate_to_mplace_no_validate(
        &mut self,
        value: Immediate<M::Provenance>,
        layout: TyAndLayout<'tcx>,
        dest: MemPlace<M::Provenance>,
    ) -> InterpResult<'tcx> {
        // We use the sizes from `value` below.
        // Ensure that matches the type of the place it is written to.
        value.assert_matches_abi(
            layout.backend_repr,
            "invalid immediate for given destination place",
            self,
        );
        // Note that it is really important that the type here is the right one, and matches the
        // type things are read at. In case `value` is a `ScalarPair`, we don't do any magic here
        // to handle padding properly, which is only correct if we never look at this data with the
        // wrong type.

        let tcx = *self.tcx;
        let Some(mut alloc) = self.get_place_alloc_mut(&MPlaceTy { mplace: dest, layout })? else {
            // zero-sized access
            return interp_ok(());
        };

        match value {
            Immediate::Scalar(scalar) => {
                alloc.write_scalar(alloc_range(Size::ZERO, scalar.size()), scalar)?;
            }
            Immediate::ScalarPair(a_val, b_val) => {
                let BackendRepr::ScalarPair(a, b) = layout.backend_repr else {
                    span_bug!(
                        self.cur_span(),
                        "write_immediate_to_mplace: invalid ScalarPair layout: {:#?}",
                        layout
                    )
                };
                let b_offset = a.size(&tcx).align_to(b.align(&tcx).abi);
                assert!(b_offset.bytes() > 0); // in `operand_field` we use the offset to tell apart the fields

                // It is tempting to verify `b_offset` against `layout.fields.offset(1)`,
                // but that does not work: We could be a newtype around a pair, then the
                // fields do not match the `ScalarPair` components.

                alloc.write_scalar(alloc_range(Size::ZERO, a_val.size()), a_val)?;
                alloc.write_scalar(alloc_range(b_offset, b_val.size()), b_val)?;
                // We don't have to reset padding here, `write_immediate` will anyway do a validation run.
            }
            Immediate::Uninit => alloc.write_uninit_full(),
        }
        interp_ok(())
    }

    pub fn write_uninit(
        &mut self,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        match self.as_mplace_or_mutable_local(&dest.to_place())? {
            Right((local_val, _local_layout, local)) => {
                *local_val = Immediate::Uninit;
                // Call the machine hook (the data race detector needs to know about this write).
                if !self.validation_in_progress() {
                    M::after_local_write(self, local, /*storage_live*/ false)?;
                }
            }
            Left(mplace) => {
                let Some(mut alloc) = self.get_place_alloc_mut(&mplace)? else {
                    // Zero-sized access
                    return interp_ok(());
                };
                alloc.write_uninit_full();
            }
        }
        interp_ok(())
    }

    /// Remove all provenance in the given place.
    pub fn clear_provenance(
        &mut self,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        // If this is an efficiently represented local variable without provenance, skip the
        // `as_mplace_or_mutable_local` that would otherwise force this local into memory.
        if let Right(imm) = dest.to_op(self)?.as_mplace_or_imm() {
            if !imm.has_provenance() {
                return interp_ok(());
            }
        }
        match self.as_mplace_or_mutable_local(&dest.to_place())? {
            Right((local_val, _local_layout, local)) => {
                local_val.clear_provenance()?;
                // Call the machine hook (the data race detector needs to know about this write).
                if !self.validation_in_progress() {
                    M::after_local_write(self, local, /*storage_live*/ false)?;
                }
            }
            Left(mplace) => {
                let Some(mut alloc) = self.get_place_alloc_mut(&mplace)? else {
                    // Zero-sized access
                    return interp_ok(());
                };
                alloc.clear_provenance();
            }
        }
        interp_ok(())
    }

    /// Copies the data from an operand to a place.
    /// The layouts of the `src` and `dest` may disagree.
    #[inline(always)]
    pub fn copy_op_allow_transmute(
        &mut self,
        src: &impl Projectable<'tcx, M::Provenance>,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.copy_op_inner(src, dest, /* allow_transmute */ true)
    }

    /// Copies the data from an operand to a place.
    /// `src` and `dest` must have the same layout and the copied value will be validated.
    #[inline(always)]
    pub fn copy_op(
        &mut self,
        src: &impl Projectable<'tcx, M::Provenance>,
        dest: &impl Writeable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.copy_op_inner(src, dest, /* allow_transmute */ false)
    }

    /// Copies the data from an operand to a place.
    /// `allow_transmute` indicates whether the layouts may disagree.
    #[inline(always)]
    #[instrument(skip(self), level = "trace")]
    fn copy_op_inner(
        &mut self,
        src: &impl Projectable<'tcx, M::Provenance>,
        dest: &impl Writeable<'tcx, M::Provenance>,
        allow_transmute: bool,
    ) -> InterpResult<'tcx> {
        // These are technically *two* typed copies: `src` is a not-yet-loaded value,
        // so we're doing a typed copy at `src` type from there to some intermediate storage.
        // And then we're doing a second typed copy from that intermediate storage to `dest`.
        // But as an optimization, we only make a single direct copy here.

        // Do the actual copy.
        self.copy_op_no_validate(src, dest, allow_transmute)?;

        if M::enforce_validity(self, dest.layout()) {
            let dest = dest.to_place();
            // Given that there were two typed copies, we have to ensure this is valid at both types,
            // and we have to ensure this loses provenance and padding according to both types.
            // But if the types are identical, we only do one pass.
            if src.layout().ty != dest.layout().ty {
                self.validate_operand(
                    &dest.transmute(src.layout(), self)?,
                    M::enforce_validity_recursively(self, src.layout()),
                    /*reset_provenance_and_padding*/ true,
                )?;
            }
            self.validate_operand(
                &dest,
                M::enforce_validity_recursively(self, dest.layout()),
                /*reset_provenance_and_padding*/ true,
            )?;
        }

        interp_ok(())
    }

    /// Copies the data from an operand to a place.
    /// `allow_transmute` indicates whether the layouts may disagree.
    /// Also, if you use this you are responsible for validating that things get copied at the
    /// right type.
    #[instrument(skip(self), level = "trace")]
    pub(super) fn copy_op_no_validate(
        &mut self,
        src: &impl Projectable<'tcx, M::Provenance>,
        dest: &impl Writeable<'tcx, M::Provenance>,
        allow_transmute: bool,
    ) -> InterpResult<'tcx> {
        // We do NOT compare the types for equality, because well-typed code can
        // actually "transmute" `&mut T` to `&T` in an assignment without a cast.
        let layout_compat =
            mir_assign_valid_types(*self.tcx, self.typing_env, src.layout(), dest.layout());
        if !allow_transmute && !layout_compat {
            span_bug!(
                self.cur_span(),
                "type mismatch when copying!\nsrc: {},\ndest: {}",
                src.layout().ty,
                dest.layout().ty,
            );
        }

        // Let us see if the layout is simple so we take a shortcut,
        // avoid force_allocation.
        let src = match self.read_immediate_raw(src)? {
            Right(src_val) => {
                assert!(!src.layout().is_unsized());
                assert!(!dest.layout().is_unsized());
                assert_eq!(src.layout().size, dest.layout().size);
                // Yay, we got a value that we can write directly.
                return if layout_compat {
                    self.write_immediate_no_validate(*src_val, dest)
                } else {
                    // This is tricky. The problematic case is `ScalarPair`: the `src_val` was
                    // loaded using the offsets defined by `src.layout`. When we put this back into
                    // the destination, we have to use the same offsets! So (a) we make sure we
                    // write back to memory, and (b) we use `dest` *with the source layout*.
                    let dest_mem = dest.force_mplace(self)?;
                    self.write_immediate_to_mplace_no_validate(
                        *src_val,
                        src.layout(),
                        dest_mem.mplace,
                    )
                };
            }
            Left(mplace) => mplace,
        };
        // Slow path, this does not fit into an immediate. Just memcpy.
        trace!("copy_op: {:?} <- {:?}: {}", *dest, src, dest.layout().ty);

        let dest = dest.force_mplace(self)?;
        let Some((dest_size, _)) = self.size_and_align_of_val(&dest)? else {
            span_bug!(self.cur_span(), "copy_op needs (dynamically) sized values")
        };
        if cfg!(debug_assertions) {
            let src_size = self.size_and_align_of_val(&src)?.unwrap().0;
            assert_eq!(src_size, dest_size, "Cannot copy differently-sized data");
        } else {
            // As a cheap approximation, we compare the fixed parts of the size.
            assert_eq!(src.layout.size, dest.layout.size);
        }

        // Setting `nonoverlapping` here only has an effect when we don't hit the fast-path above,
        // but that should at least match what LLVM does where `memcpy` is also only used when the
        // type does not have Scalar/ScalarPair layout.
        // (Or as the `Assign` docs put it, assignments "not producing primitives" must be
        // non-overlapping.)
        // We check alignment separately, and *after* checking everything else.
        // If an access is both OOB and misaligned, we want to see the bounds error.
        self.mem_copy(src.ptr(), dest.ptr(), dest_size, /*nonoverlapping*/ true)?;
        self.check_misalign(src.mplace.misaligned, CheckAlignMsg::BasedOn)?;
        self.check_misalign(dest.mplace.misaligned, CheckAlignMsg::BasedOn)?;
        interp_ok(())
    }

    /// Ensures that a place is in memory, and returns where it is.
    /// If the place currently refers to a local that doesn't yet have a matching allocation,
    /// create such an allocation.
    /// This is essentially `force_to_memplace`.
    #[instrument(skip(self), level = "trace")]
    pub fn force_allocation(
        &mut self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let mplace = match place.place {
            Place::Local { local, offset, locals_addr } => {
                debug_assert_eq!(locals_addr, self.frame().locals_addr());
                let whole_local = match self.frame_mut().locals[local].access_mut()? {
                    &mut Operand::Immediate(local_val) => {
                        // We need to make an allocation.

                        // We need the layout of the local. We can NOT use the layout we got,
                        // that might e.g., be an inner field of a struct with `Scalar` layout,
                        // that has different alignment than the outer field.
                        let local_layout = self.layout_of_local(&self.frame(), local, None)?;
                        assert!(local_layout.is_sized(), "unsized locals cannot be immediate");
                        let mplace = self.allocate(local_layout, MemoryKind::Stack)?;
                        // Preserve old value. (As an optimization, we can skip this if it was uninit.)
                        if !matches!(local_val, Immediate::Uninit) {
                            // We don't have to validate as we can assume the local was already
                            // valid for its type. We must not use any part of `place` here, that
                            // could be a projection to a part of the local!
                            self.write_immediate_to_mplace_no_validate(
                                local_val,
                                local_layout,
                                mplace.mplace,
                            )?;
                        }
                        M::after_local_moved_to_memory(self, local, &mplace)?;
                        // Now we can call `access_mut` again, asserting it goes well, and actually
                        // overwrite things. This points to the entire allocation, not just the part
                        // the place refers to, i.e. we do this before we apply `offset`.
                        *self.frame_mut().locals[local].access_mut().unwrap() =
                            Operand::Indirect(mplace.mplace);
                        mplace.mplace
                    }
                    &mut Operand::Indirect(mplace) => mplace, // this already was an indirect local
                };
                if let Some(offset) = offset {
                    // This offset is always inbounds, no need to check it again.
                    whole_local.offset_with_meta_(
                        offset,
                        OffsetMode::Wrapping,
                        MemPlaceMeta::None,
                        self,
                    )?
                } else {
                    // Preserve wide place metadata, do not call `offset`.
                    whole_local
                }
            }
            Place::Ptr(mplace) => mplace,
        };
        // Return with the original layout and align, so that the caller can go on
        interp_ok(MPlaceTy { mplace, layout: place.layout })
    }

    pub fn allocate_dyn(
        &mut self,
        layout: TyAndLayout<'tcx>,
        kind: MemoryKind<M::MemoryKind>,
        meta: MemPlaceMeta<M::Provenance>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let Some((size, align)) = self.size_and_align_from_meta(&meta, &layout)? else {
            span_bug!(self.cur_span(), "cannot allocate space for `extern` type, size is not known")
        };
        let ptr = self.allocate_ptr(size, align, kind, AllocInit::Uninit)?;
        interp_ok(self.ptr_with_meta_to_mplace(ptr.into(), meta, layout, /*unaligned*/ false))
    }

    pub fn allocate(
        &mut self,
        layout: TyAndLayout<'tcx>,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        assert!(layout.is_sized());
        self.allocate_dyn(layout, kind, MemPlaceMeta::None)
    }

    /// Allocates a sequence of bytes in the interpreter's memory with alignment 1.
    /// This is allocated in immutable global memory and deduplicated.
    pub fn allocate_bytes_dedup(
        &mut self,
        bytes: &[u8],
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let salt = M::get_global_alloc_salt(self, None);
        let id = self.tcx.allocate_bytes_dedup(bytes, salt);

        // Turn untagged "global" pointers (obtained via `tcx`) into the machine pointer to the allocation.
        M::adjust_alloc_root_pointer(
            &self,
            Pointer::from(id),
            M::GLOBAL_KIND.map(MemoryKind::Machine),
        )
    }

    /// Allocates a string in the interpreter's memory, returning it as a (wide) place.
    /// This is allocated in immutable global memory and deduplicated.
    pub fn allocate_str_dedup(
        &mut self,
        s: &str,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let bytes = s.as_bytes();
        let ptr = self.allocate_bytes_dedup(bytes)?;

        // Create length metadata for the string.
        let meta = Scalar::from_target_usize(u64::try_from(bytes.len()).unwrap(), self);

        // Get layout for Rust's str type.
        let layout = self.layout_of(self.tcx.types.str_).unwrap();

        // Combine pointer and metadata into a wide pointer.
        interp_ok(self.ptr_with_meta_to_mplace(
            ptr.into(),
            MemPlaceMeta::Meta(meta),
            layout,
            /*unaligned*/ false,
        ))
    }

    pub fn raw_const_to_mplace(
        &self,
        raw: mir::ConstAlloc<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        // This must be an allocation in `tcx`
        let _ = self.tcx.global_alloc(raw.alloc_id);
        let ptr = self.global_root_pointer(Pointer::from(raw.alloc_id))?;
        let layout = self.layout_of(raw.ty)?;
        interp_ok(self.ptr_to_mplace(ptr.into(), layout))
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(MPlaceTy<'_>, 64);
    static_assert_size!(MemPlace, 48);
    static_assert_size!(MemPlaceMeta, 24);
    static_assert_size!(Place, 48);
    static_assert_size!(PlaceTy<'_>, 64);
    // tidy-alphabetical-end
}
