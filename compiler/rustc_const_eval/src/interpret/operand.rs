//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use either::{Either, Left, Right};

use rustc_hir::def::Namespace;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::print::{FmtPrinter, PrettyPrinter};
use rustc_middle::ty::{ConstInt, Ty, ValTree};
use rustc_middle::{mir, ty};
use rustc_span::Span;
use rustc_target::abi::{self, Abi, Align, HasDataLayout, Size};

use super::{
    alloc_range, from_known_layout, mir_assign_valid_types, AllocId, ConstValue, Frame, GlobalId,
    InterpCx, InterpResult, MPlaceTy, Machine, MemPlace, MemPlaceMeta, Place, PlaceTy, Pointer,
    Provenance, Scalar,
};

/// An `Immediate` represents a single immediate self-contained Rust value.
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and wide pointers. This idea was taken from rustc's codegen.
/// In particular, thanks to `ScalarPair`, arithmetic operations and casts can be entirely
/// defined on `Immediate`, and do not have to work with a `Place`.
#[derive(Copy, Clone, Debug)]
pub enum Immediate<Prov: Provenance = AllocId> {
    /// A single scalar value (must have *initialized* `Scalar` ABI).
    Scalar(Scalar<Prov>),
    /// A pair of two scalar value (must have `ScalarPair` ABI where both fields are
    /// `Scalar::Initialized`).
    ScalarPair(Scalar<Prov>, Scalar<Prov>),
    /// A value of fully uninitialized memory. Can have and size and layout.
    Uninit,
}

impl<Prov: Provenance> From<Scalar<Prov>> for Immediate<Prov> {
    #[inline(always)]
    fn from(val: Scalar<Prov>) -> Self {
        Immediate::Scalar(val)
    }
}

impl<Prov: Provenance> Immediate<Prov> {
    pub fn from_pointer(p: Pointer<Prov>, cx: &impl HasDataLayout) -> Self {
        Immediate::Scalar(Scalar::from_pointer(p, cx))
    }

    pub fn from_maybe_pointer(p: Pointer<Option<Prov>>, cx: &impl HasDataLayout) -> Self {
        Immediate::Scalar(Scalar::from_maybe_pointer(p, cx))
    }

    pub fn new_slice(val: Scalar<Prov>, len: u64, cx: &impl HasDataLayout) -> Self {
        Immediate::ScalarPair(val, Scalar::from_target_usize(len, cx))
    }

    pub fn new_dyn_trait(
        val: Scalar<Prov>,
        vtable: Pointer<Option<Prov>>,
        cx: &impl HasDataLayout,
    ) -> Self {
        Immediate::ScalarPair(val, Scalar::from_maybe_pointer(vtable, cx))
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn to_scalar(self) -> Scalar<Prov> {
        match self {
            Immediate::Scalar(val) => val,
            Immediate::ScalarPair(..) => bug!("Got a scalar pair where a scalar was expected"),
            Immediate::Uninit => bug!("Got uninit where a scalar was expected"),
        }
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn to_scalar_pair(self) -> (Scalar<Prov>, Scalar<Prov>) {
        match self {
            Immediate::ScalarPair(val1, val2) => (val1, val2),
            Immediate::Scalar(..) => bug!("Got a scalar where a scalar pair was expected"),
            Immediate::Uninit => bug!("Got uninit where a scalar pair was expected"),
        }
    }
}

// ScalarPair needs a type to interpret, so we often have an immediate and a type together
// as input for binary and cast operations.
#[derive(Clone, Debug)]
pub struct ImmTy<'tcx, Prov: Provenance = AllocId> {
    imm: Immediate<Prov>,
    pub layout: TyAndLayout<'tcx>,
}

impl<Prov: Provenance> std::fmt::Display for ImmTy<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Helper function for printing a scalar to a FmtPrinter
        fn p<'a, 'tcx, Prov: Provenance>(
            cx: FmtPrinter<'a, 'tcx>,
            s: Scalar<Prov>,
            ty: Ty<'tcx>,
        ) -> Result<FmtPrinter<'a, 'tcx>, std::fmt::Error> {
            match s {
                Scalar::Int(int) => cx.pretty_print_const_scalar_int(int, ty, true),
                Scalar::Ptr(ptr, _sz) => {
                    // Just print the ptr value. `pretty_print_const_scalar_ptr` would also try to
                    // print what is points to, which would fail since it has no access to the local
                    // memory.
                    cx.pretty_print_const_pointer(ptr, ty)
                }
            }
        }
        ty::tls::with(|tcx| {
            match self.imm {
                Immediate::Scalar(s) => {
                    if let Some(ty) = tcx.lift(self.layout.ty) {
                        let cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                        f.write_str(&p(cx, s, ty)?.into_buffer())?;
                        return Ok(());
                    }
                    write!(f, "{:x}: {}", s, self.layout.ty)
                }
                Immediate::ScalarPair(a, b) => {
                    // FIXME(oli-obk): at least print tuples and slices nicely
                    write!(f, "({:x}, {:x}): {}", a, b, self.layout.ty)
                }
                Immediate::Uninit => {
                    write!(f, "uninit: {}", self.layout.ty)
                }
            }
        })
    }
}

impl<'tcx, Prov: Provenance> std::ops::Deref for ImmTy<'tcx, Prov> {
    type Target = Immediate<Prov>;
    #[inline(always)]
    fn deref(&self) -> &Immediate<Prov> {
        &self.imm
    }
}

/// An `Operand` is the result of computing a `mir::Operand`. It can be immediate,
/// or still in memory. The latter is an optimization, to delay reading that chunk of
/// memory and to avoid having to store arbitrary-sized data here.
#[derive(Copy, Clone, Debug)]
pub enum Operand<Prov: Provenance = AllocId> {
    Immediate(Immediate<Prov>),
    Indirect(MemPlace<Prov>),
}

#[derive(Clone, Debug)]
pub struct OpTy<'tcx, Prov: Provenance = AllocId> {
    op: Operand<Prov>, // Keep this private; it helps enforce invariants.
    pub layout: TyAndLayout<'tcx>,
    /// rustc does not have a proper way to represent the type of a field of a `repr(packed)` struct:
    /// it needs to have a different alignment than the field type would usually have.
    /// So we represent this here with a separate field that "overwrites" `layout.align`.
    /// This means `layout.align` should never be used for an `OpTy`!
    /// `None` means "alignment does not matter since this is a by-value operand"
    /// (`Operand::Immediate`); this field is only relevant for `Operand::Indirect`.
    /// Also CTFE ignores alignment anyway, so this is for Miri only.
    pub align: Option<Align>,
}

impl<'tcx, Prov: Provenance> std::ops::Deref for OpTy<'tcx, Prov> {
    type Target = Operand<Prov>;
    #[inline(always)]
    fn deref(&self) -> &Operand<Prov> {
        &self.op
    }
}

impl<'tcx, Prov: Provenance> From<MPlaceTy<'tcx, Prov>> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: MPlaceTy<'tcx, Prov>) -> Self {
        OpTy { op: Operand::Indirect(*mplace), layout: mplace.layout, align: Some(mplace.align) }
    }
}

impl<'tcx, Prov: Provenance> From<&'_ MPlaceTy<'tcx, Prov>> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: &MPlaceTy<'tcx, Prov>) -> Self {
        OpTy { op: Operand::Indirect(**mplace), layout: mplace.layout, align: Some(mplace.align) }
    }
}

impl<'tcx, Prov: Provenance> From<&'_ mut MPlaceTy<'tcx, Prov>> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: &mut MPlaceTy<'tcx, Prov>) -> Self {
        OpTy { op: Operand::Indirect(**mplace), layout: mplace.layout, align: Some(mplace.align) }
    }
}

impl<'tcx, Prov: Provenance> From<ImmTy<'tcx, Prov>> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn from(val: ImmTy<'tcx, Prov>) -> Self {
        OpTy { op: Operand::Immediate(val.imm), layout: val.layout, align: None }
    }
}

impl<'tcx, Prov: Provenance> ImmTy<'tcx, Prov> {
    #[inline]
    pub fn from_scalar(val: Scalar<Prov>, layout: TyAndLayout<'tcx>) -> Self {
        ImmTy { imm: val.into(), layout }
    }

    #[inline]
    pub fn from_immediate(imm: Immediate<Prov>, layout: TyAndLayout<'tcx>) -> Self {
        ImmTy { imm, layout }
    }

    #[inline]
    pub fn uninit(layout: TyAndLayout<'tcx>) -> Self {
        ImmTy { imm: Immediate::Uninit, layout }
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, layout: TyAndLayout<'tcx>) -> Option<Self> {
        Some(Self::from_scalar(Scalar::try_from_uint(i, layout.size)?, layout))
    }
    #[inline]
    pub fn from_uint(i: impl Into<u128>, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_uint(i, layout.size), layout)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, layout: TyAndLayout<'tcx>) -> Option<Self> {
        Some(Self::from_scalar(Scalar::try_from_int(i, layout.size)?, layout))
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_int(i, layout.size), layout)
    }

    #[inline]
    pub fn to_const_int(self) -> ConstInt {
        assert!(self.layout.ty.is_integral());
        let int = self.to_scalar().assert_int();
        ConstInt::new(int, self.layout.ty.is_signed(), self.layout.ty.is_ptr_sized_integral())
    }
}

impl<'tcx, Prov: Provenance> OpTy<'tcx, Prov> {
    pub fn len(&self, cx: &impl HasDataLayout) -> InterpResult<'tcx, u64> {
        if self.layout.is_unsized() {
            if matches!(self.op, Operand::Immediate(Immediate::Uninit)) {
                // Uninit unsized places shouldn't occur. In the interpreter we have them
                // temporarily for unsized arguments before their value is put in; in ConstProp they
                // remain uninit and this code can actually be reached.
                throw_inval!(UninitUnsizedLocal);
            }
            // There are no unsized immediates.
            self.assert_mem_place().len(cx)
        } else {
            match self.layout.fields {
                abi::FieldsShape::Array { count, .. } => Ok(count),
                _ => bug!("len not supported on sized type {:?}", self.layout.ty),
            }
        }
    }

    /// Replace the layout of this operand. There's basically no sanity check that this makes sense,
    /// you better know what you are doing! If this is an immediate, applying the wrong layout can
    /// not just lead to invalid data, it can actually *shift the data around* since the offsets of
    /// a ScalarPair are entirely determined by the layout, not the data.
    pub fn transmute(&self, layout: TyAndLayout<'tcx>) -> Self {
        assert_eq!(
            self.layout.size, layout.size,
            "transmuting with a size change, that doesn't seem right"
        );
        OpTy { layout, ..*self }
    }

    /// Offset the operand in memory (if possible) and change its metadata.
    ///
    /// This can go wrong very easily if you give the wrong layout for the new place!
    pub(super) fn offset_with_meta(
        &self,
        offset: Size,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Self> {
        match self.as_mplace_or_imm() {
            Left(mplace) => Ok(mplace.offset_with_meta(offset, meta, layout, cx)?.into()),
            Right(imm) => {
                assert!(
                    matches!(*imm, Immediate::Uninit),
                    "Scalar/ScalarPair cannot be offset into"
                );
                assert!(!meta.has_meta()); // no place to store metadata here
                // Every part of an uninit is uninit.
                Ok(ImmTy::uninit(layout).into())
            }
        }
    }

    /// Offset the operand in memory (if possible).
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
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Try reading an immediate in memory; this is interesting particularly for `ScalarPair`.
    /// Returns `None` if the layout does not permit loading this as a value.
    ///
    /// This is an internal function; call `read_immediate` instead.
    fn read_immediate_from_mplace_raw(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Option<ImmTy<'tcx, M::Provenance>>> {
        if mplace.layout.is_unsized() {
            // Don't touch unsized
            return Ok(None);
        }

        let Some(alloc) = self.get_place_alloc(mplace)? else {
            // zero-sized type can be left uninit
            return Ok(Some(ImmTy::uninit(mplace.layout)));
        };

        // It may seem like all types with `Scalar` or `ScalarPair` ABI are fair game at this point.
        // However, `MaybeUninit<u64>` is considered a `Scalar` as far as its layout is concerned --
        // and yet cannot be represented by an interpreter `Scalar`, since we have to handle the
        // case where some of the bytes are initialized and others are not. So, we need an extra
        // check that walks over the type of `mplace` to make sure it is truly correct to treat this
        // like a `Scalar` (or `ScalarPair`).
        Ok(match mplace.layout.abi {
            Abi::Scalar(abi::Scalar::Initialized { value: s, .. }) => {
                let size = s.size(self);
                assert_eq!(size, mplace.layout.size, "abi::Scalar size does not match layout size");
                let scalar = alloc.read_scalar(
                    alloc_range(Size::ZERO, size),
                    /*read_provenance*/ matches!(s, abi::Pointer(_)),
                )?;
                Some(ImmTy { imm: scalar.into(), layout: mplace.layout })
            }
            Abi::ScalarPair(
                abi::Scalar::Initialized { value: a, .. },
                abi::Scalar::Initialized { value: b, .. },
            ) => {
                // We checked `ptr_align` above, so all fields will have the alignment they need.
                // We would anyway check against `ptr_align.restrict_for_offset(b_offset)`,
                // which `ptr.offset(b_offset)` cannot possibly fail to satisfy.
                let (a_size, b_size) = (a.size(self), b.size(self));
                let b_offset = a_size.align_to(b.align(self).abi);
                assert!(b_offset.bytes() > 0); // in `operand_field` we use the offset to tell apart the fields
                let a_val = alloc.read_scalar(
                    alloc_range(Size::ZERO, a_size),
                    /*read_provenance*/ matches!(a, abi::Pointer(_)),
                )?;
                let b_val = alloc.read_scalar(
                    alloc_range(b_offset, b_size),
                    /*read_provenance*/ matches!(b, abi::Pointer(_)),
                )?;
                Some(ImmTy { imm: Immediate::ScalarPair(a_val, b_val), layout: mplace.layout })
            }
            _ => {
                // Neither a scalar nor scalar pair.
                None
            }
        })
    }

    /// Try returning an immediate for the operand. If the layout does not permit loading this as an
    /// immediate, return where in memory we can find the data.
    /// Note that for a given layout, this operation will either always return Left or Right!
    /// succeed!  Whether it returns Left depends on whether the layout can be represented
    /// in an `Immediate`, not on which data is stored there currently.
    ///
    /// This is an internal function that should not usually be used; call `read_immediate` instead.
    /// ConstProp needs it, though.
    pub fn read_immediate_raw(
        &self,
        src: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Either<MPlaceTy<'tcx, M::Provenance>, ImmTy<'tcx, M::Provenance>>> {
        Ok(match src.as_mplace_or_imm() {
            Left(ref mplace) => {
                if let Some(val) = self.read_immediate_from_mplace_raw(mplace)? {
                    Right(val)
                } else {
                    Left(*mplace)
                }
            }
            Right(val) => Right(val),
        })
    }

    /// Read an immediate from a place, asserting that that is possible with the given layout.
    ///
    /// If this succeeds, the `ImmTy` is never `Uninit`.
    #[inline(always)]
    pub fn read_immediate(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        if !matches!(
            op.layout.abi,
            Abi::Scalar(abi::Scalar::Initialized { .. })
                | Abi::ScalarPair(abi::Scalar::Initialized { .. }, abi::Scalar::Initialized { .. })
        ) {
            span_bug!(self.cur_span(), "primitive read not possible for type: {:?}", op.layout.ty);
        }
        let imm = self.read_immediate_raw(op)?.right().unwrap();
        if matches!(*imm, Immediate::Uninit) {
            throw_ub!(InvalidUninitBytes(None));
        }
        Ok(imm)
    }

    /// Read a scalar from a place
    pub fn read_scalar(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        Ok(self.read_immediate(op)?.to_scalar())
    }

    // Pointer-sized reads are fairly common and need target layout access, so we wrap them in
    // convenience functions.

    /// Read a pointer from a place.
    pub fn read_pointer(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Pointer<Option<M::Provenance>>> {
        self.read_scalar(op)?.to_pointer(self)
    }
    /// Read a pointer-sized unsigned integer from a place.
    pub fn read_target_usize(&self, op: &OpTy<'tcx, M::Provenance>) -> InterpResult<'tcx, u64> {
        self.read_scalar(op)?.to_target_usize(self)
    }
    /// Read a pointer-sized signed integer from a place.
    pub fn read_target_isize(&self, op: &OpTy<'tcx, M::Provenance>) -> InterpResult<'tcx, i64> {
        self.read_scalar(op)?.to_target_isize(self)
    }

    /// Turn the wide MPlace into a string (must already be dereferenced!)
    pub fn read_str(&self, mplace: &MPlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx, &str> {
        let len = mplace.len(self)?;
        let bytes = self.read_bytes_ptr_strip_provenance(mplace.ptr, Size::from_bytes(len))?;
        let str = std::str::from_utf8(bytes).map_err(|err| err_ub!(InvalidStr(err)))?;
        Ok(str)
    }

    /// Converts a repr(simd) operand into an operand where `place_index` accesses the SIMD elements.
    /// Also returns the number of elements.
    ///
    /// Can (but does not always) trigger UB if `op` is uninitialized.
    pub fn operand_to_simd(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (MPlaceTy<'tcx, M::Provenance>, u64)> {
        // Basically we just transmute this place into an array following simd_size_and_type.
        // This only works in memory, but repr(simd) types should never be immediates anyway.
        assert!(op.layout.ty.is_simd());
        match op.as_mplace_or_imm() {
            Left(mplace) => self.mplace_to_simd(&mplace),
            Right(imm) => match *imm {
                Immediate::Uninit => {
                    throw_ub!(InvalidUninitBytes(None))
                }
                Immediate::Scalar(..) | Immediate::ScalarPair(..) => {
                    bug!("arrays/slices can never have Scalar/ScalarPair layout")
                }
            },
        }
    }

    /// Read from a local.
    /// Will not access memory, instead an indirect `Operand` is returned.
    ///
    /// This is public because it is used by [priroda](https://github.com/oli-obk/priroda) to get an
    /// OpTy from a local.
    pub fn local_to_op(
        &self,
        frame: &Frame<'mir, 'tcx, M::Provenance, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        let layout = self.layout_of_local(frame, local, layout)?;
        let op = *frame.locals[local].access()?;
        Ok(OpTy { op, layout, align: Some(layout.align.abi) })
    }

    /// Every place can be read from, so we can turn them into an operand.
    /// This will definitely return `Indirect` if the place is a `Ptr`, i.e., this
    /// will never actually read from memory.
    #[inline(always)]
    pub fn place_to_op(
        &self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        let op = match **place {
            Place::Ptr(mplace) => Operand::Indirect(mplace),
            Place::Local { frame, local } => {
                *self.local_to_op(&self.stack()[frame], local, None)?
            }
        };
        Ok(OpTy { op, layout: place.layout, align: Some(place.align) })
    }

    /// Evaluate a place with the goal of reading from it. This lets us sometimes
    /// avoid allocations.
    pub fn eval_place_to_op(
        &self,
        mir_place: mir::Place<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // Do not use the layout passed in as argument if the base we are looking at
        // here is not the entire place.
        let layout = if mir_place.projection.is_empty() { layout } else { None };

        let mut op = self.local_to_op(self.frame(), mir_place.local, layout)?;
        // Using `try_fold` turned out to be bad for performance, hence the loop.
        for elem in mir_place.projection.iter() {
            op = self.operand_projection(&op, elem)?
        }

        trace!("eval_place_to_op: got {:?}", *op);
        // Sanity-check the type we ended up with.
        debug_assert!(
            mir_assign_valid_types(
                *self.tcx,
                self.param_env,
                self.layout_of(self.subst_from_current_frame_and_normalize_erasing_regions(
                    mir_place.ty(&self.frame().body.local_decls, *self.tcx).ty
                )?)?,
                op.layout,
            ),
            "eval_place of a MIR place with type {:?} produced an interpreter operand with type {:?}",
            mir_place.ty(&self.frame().body.local_decls, *self.tcx).ty,
            op.layout.ty,
        );
        Ok(op)
    }

    /// Evaluate the operand, returning a place where you can then find the data.
    /// If you already know the layout, you can save two table lookups
    /// by passing it in here.
    #[inline]
    pub fn eval_operand(
        &self,
        mir_op: &mir::Operand<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::Operand::*;
        let op = match mir_op {
            // FIXME: do some more logic on `move` to invalidate the old location
            &Copy(place) | &Move(place) => self.eval_place_to_op(place, layout)?,

            Constant(constant) => {
                let c =
                    self.subst_from_current_frame_and_normalize_erasing_regions(constant.literal)?;

                // This can still fail:
                // * During ConstProp, with `TooGeneric` or since the `required_consts` were not all
                //   checked yet.
                // * During CTFE, since promoteds in `const`/`static` initializer bodies can fail.
                self.eval_mir_constant(&c, Some(constant.span), layout)?
            }
        };
        trace!("{:?}: {:?}", mir_op, *op);
        Ok(op)
    }

    fn eval_ty_constant(
        &self,
        val: ty::Const<'tcx>,
        span: Option<Span>,
    ) -> InterpResult<'tcx, ValTree<'tcx>> {
        Ok(match val.kind() {
            ty::ConstKind::Param(_) | ty::ConstKind::Placeholder(..) => {
                throw_inval!(TooGeneric)
            }
            // FIXME(generic_const_exprs): `ConstKind::Expr` should be able to be evaluated
            ty::ConstKind::Expr(_) => throw_inval!(TooGeneric),
            ty::ConstKind::Error(reported) => {
                throw_inval!(AlreadyReported(reported.into()))
            }
            ty::ConstKind::Unevaluated(uv) => {
                let instance = self.resolve(uv.def, uv.args)?;
                let cid = GlobalId { instance, promoted: None };
                self.ctfe_query(span, |tcx| {
                    tcx.eval_to_valtree(self.param_env.with_const().and(cid))
                })?
                .unwrap_or_else(|| bug!("unable to create ValTree for {uv:?}"))
            }
            ty::ConstKind::Bound(..) | ty::ConstKind::Infer(..) => {
                span_bug!(self.cur_span(), "unexpected ConstKind in ctfe: {val:?}")
            }
            ty::ConstKind::Value(valtree) => valtree,
        })
    }

    pub fn eval_mir_constant(
        &self,
        val: &mir::ConstantKind<'tcx>,
        span: Option<Span>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        match *val {
            mir::ConstantKind::Ty(ct) => {
                let ty = ct.ty();
                let valtree = self.eval_ty_constant(ct, span)?;
                let const_val = self.tcx.valtree_to_const_val((ty, valtree));
                self.const_val_to_op(const_val, ty, layout)
            }
            mir::ConstantKind::Val(val, ty) => self.const_val_to_op(val, ty, layout),
            mir::ConstantKind::Unevaluated(uv, _) => {
                let instance = self.resolve(uv.def, uv.args)?;
                Ok(self.eval_global(GlobalId { instance, promoted: uv.promoted }, span)?.into())
            }
        }
    }

    pub(crate) fn const_val_to_op(
        &self,
        val_val: ConstValue<'tcx>,
        ty: Ty<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // Other cases need layout.
        let adjust_scalar = |scalar| -> InterpResult<'tcx, _> {
            Ok(match scalar {
                Scalar::Ptr(ptr, size) => Scalar::Ptr(self.global_base_pointer(ptr)?, size),
                Scalar::Int(int) => Scalar::Int(int),
            })
        };
        let layout = from_known_layout(self.tcx, self.param_env, layout, || self.layout_of(ty))?;
        let op = match val_val {
            ConstValue::ByRef { alloc, offset } => {
                let id = self.tcx.create_memory_alloc(alloc);
                // We rely on mutability being set correctly in that allocation to prevent writes
                // where none should happen.
                let ptr = self.global_base_pointer(Pointer::new(id, offset))?;
                Operand::Indirect(MemPlace::from_ptr(ptr.into()))
            }
            ConstValue::Scalar(x) => Operand::Immediate(adjust_scalar(x)?.into()),
            ConstValue::ZeroSized => Operand::Immediate(Immediate::Uninit),
            ConstValue::Slice { data, start, end } => {
                // We rely on mutability being set correctly in `data` to prevent writes
                // where none should happen.
                let ptr = Pointer::new(
                    self.tcx.create_memory_alloc(data),
                    Size::from_bytes(start), // offset: `start`
                );
                Operand::Immediate(Immediate::new_slice(
                    Scalar::from_pointer(self.global_base_pointer(ptr)?, &*self.tcx),
                    u64::try_from(end.checked_sub(start).unwrap()).unwrap(), // len: `end - start`
                    self,
                ))
            }
        };
        Ok(OpTy { op, layout, align: Some(layout.align.abi) })
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(Immediate, 48);
    static_assert_size!(ImmTy<'_>, 64);
    static_assert_size!(Operand, 56);
    static_assert_size!(OpTy<'_>, 80);
    // tidy-alphabetical-end
}
