//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use std::assert_matches::assert_matches;

use either::{Either, Left, Right};
use rustc_abi as abi;
use rustc_abi::{BackendRepr, HasDataLayout, Size};
use rustc_hir::def::Namespace;
use rustc_middle::mir::interpret::ScalarSizeMismatch;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, TyAndLayout};
use rustc_middle::ty::print::{FmtPrinter, PrettyPrinter};
use rustc_middle::ty::{ConstInt, ScalarInt, Ty, TyCtxt};
use rustc_middle::{bug, mir, span_bug, ty};
use rustc_span::DUMMY_SP;
use tracing::trace;

use super::{
    CtfeProvenance, Frame, InterpCx, InterpResult, MPlaceTy, Machine, MemPlace, MemPlaceMeta,
    OffsetMode, PlaceTy, Pointer, Projectable, Provenance, Scalar, alloc_range, err_ub,
    from_known_layout, interp_ok, mir_assign_valid_types, throw_ub,
};

/// An `Immediate` represents a single immediate self-contained Rust value.
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and wide pointers. This idea was taken from rustc's codegen.
/// In particular, thanks to `ScalarPair`, arithmetic operations and casts can be entirely
/// defined on `Immediate`, and do not have to work with a `Place`.
#[derive(Copy, Clone, Debug)]
pub enum Immediate<Prov: Provenance = CtfeProvenance> {
    /// A single scalar value (must have *initialized* `Scalar` ABI).
    Scalar(Scalar<Prov>),
    /// A pair of two scalar value (must have `ScalarPair` ABI where both fields are
    /// `Scalar::Initialized`).
    ScalarPair(Scalar<Prov>, Scalar<Prov>),
    /// A value of fully uninitialized memory. Can have arbitrary size and layout, but must be sized.
    Uninit,
}

impl<Prov: Provenance> From<Scalar<Prov>> for Immediate<Prov> {
    #[inline(always)]
    fn from(val: Scalar<Prov>) -> Self {
        Immediate::Scalar(val)
    }
}

impl<Prov: Provenance> Immediate<Prov> {
    pub fn new_pointer_with_meta(
        ptr: Pointer<Option<Prov>>,
        meta: MemPlaceMeta<Prov>,
        cx: &impl HasDataLayout,
    ) -> Self {
        let ptr = Scalar::from_maybe_pointer(ptr, cx);
        match meta {
            MemPlaceMeta::None => Immediate::from(ptr),
            MemPlaceMeta::Meta(meta) => Immediate::ScalarPair(ptr, meta),
        }
    }

    pub fn new_slice(ptr: Pointer<Option<Prov>>, len: u64, cx: &impl HasDataLayout) -> Self {
        Immediate::ScalarPair(
            Scalar::from_maybe_pointer(ptr, cx),
            Scalar::from_target_usize(len, cx),
        )
    }

    pub fn new_dyn_trait(
        val: Pointer<Option<Prov>>,
        vtable: Pointer<Option<Prov>>,
        cx: &impl HasDataLayout,
    ) -> Self {
        Immediate::ScalarPair(
            Scalar::from_maybe_pointer(val, cx),
            Scalar::from_maybe_pointer(vtable, cx),
        )
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
    pub fn to_scalar_int(self) -> ScalarInt {
        self.to_scalar().try_to_scalar_int().unwrap()
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

    /// Returns the scalar from the first component and optionally the 2nd component as metadata.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn to_scalar_and_meta(self) -> (Scalar<Prov>, MemPlaceMeta<Prov>) {
        match self {
            Immediate::ScalarPair(val1, val2) => (val1, MemPlaceMeta::Meta(val2)),
            Immediate::Scalar(val) => (val, MemPlaceMeta::None),
            Immediate::Uninit => bug!("Got uninit where a scalar or scalar pair was expected"),
        }
    }

    /// Assert that this immediate is a valid value for the given ABI.
    pub fn assert_matches_abi(self, abi: BackendRepr, msg: &str, cx: &impl HasDataLayout) {
        match (self, abi) {
            (Immediate::Scalar(scalar), BackendRepr::Scalar(s)) => {
                assert_eq!(scalar.size(), s.size(cx), "{msg}: scalar value has wrong size");
                if !matches!(s.primitive(), abi::Primitive::Pointer(..)) {
                    // This is not a pointer, it should not carry provenance.
                    assert!(
                        matches!(scalar, Scalar::Int(..)),
                        "{msg}: scalar value should be an integer, but has provenance"
                    );
                }
            }
            (Immediate::ScalarPair(a_val, b_val), BackendRepr::ScalarPair(a, b)) => {
                assert_eq!(
                    a_val.size(),
                    a.size(cx),
                    "{msg}: first component of scalar pair has wrong size"
                );
                if !matches!(a.primitive(), abi::Primitive::Pointer(..)) {
                    assert!(
                        matches!(a_val, Scalar::Int(..)),
                        "{msg}: first component of scalar pair should be an integer, but has provenance"
                    );
                }
                assert_eq!(
                    b_val.size(),
                    b.size(cx),
                    "{msg}: second component of scalar pair has wrong size"
                );
                if !matches!(b.primitive(), abi::Primitive::Pointer(..)) {
                    assert!(
                        matches!(b_val, Scalar::Int(..)),
                        "{msg}: second component of scalar pair should be an integer, but has provenance"
                    );
                }
            }
            (Immediate::Uninit, _) => {
                assert!(abi.is_sized(), "{msg}: unsized immediates are not a thing");
            }
            _ => {
                bug!("{msg}: value {self:?} does not match ABI {abi:?})",)
            }
        }
    }

    pub fn clear_provenance<'tcx>(&mut self) -> InterpResult<'tcx> {
        match self {
            Immediate::Scalar(s) => {
                s.clear_provenance()?;
            }
            Immediate::ScalarPair(a, b) => {
                a.clear_provenance()?;
                b.clear_provenance()?;
            }
            Immediate::Uninit => {}
        }
        interp_ok(())
    }
}

// ScalarPair needs a type to interpret, so we often have an immediate and a type together
// as input for binary and cast operations.
#[derive(Clone)]
pub struct ImmTy<'tcx, Prov: Provenance = CtfeProvenance> {
    imm: Immediate<Prov>,
    pub layout: TyAndLayout<'tcx>,
}

impl<Prov: Provenance> std::fmt::Display for ImmTy<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Helper function for printing a scalar to a FmtPrinter
        fn p<'a, 'tcx, Prov: Provenance>(
            cx: &mut FmtPrinter<'a, 'tcx>,
            s: Scalar<Prov>,
            ty: Ty<'tcx>,
        ) -> Result<(), std::fmt::Error> {
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
                        let s =
                            FmtPrinter::print_string(tcx, Namespace::ValueNS, |cx| p(cx, s, ty))?;
                        f.write_str(&s)?;
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

impl<Prov: Provenance> std::fmt::Debug for ImmTy<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Printing `layout` results in too much noise; just print a nice version of the type.
        f.debug_struct("ImmTy")
            .field("imm", &self.imm)
            .field("ty", &format_args!("{}", self.layout.ty))
            .finish()
    }
}

impl<'tcx, Prov: Provenance> std::ops::Deref for ImmTy<'tcx, Prov> {
    type Target = Immediate<Prov>;
    #[inline(always)]
    fn deref(&self) -> &Immediate<Prov> {
        &self.imm
    }
}

impl<'tcx, Prov: Provenance> ImmTy<'tcx, Prov> {
    #[inline]
    pub fn from_scalar(val: Scalar<Prov>, layout: TyAndLayout<'tcx>) -> Self {
        debug_assert!(layout.backend_repr.is_scalar(), "`ImmTy::from_scalar` on non-scalar layout");
        debug_assert_eq!(val.size(), layout.size);
        ImmTy { imm: val.into(), layout }
    }

    #[inline]
    pub fn from_scalar_pair(a: Scalar<Prov>, b: Scalar<Prov>, layout: TyAndLayout<'tcx>) -> Self {
        debug_assert!(
            matches!(layout.backend_repr, BackendRepr::ScalarPair(..)),
            "`ImmTy::from_scalar_pair` on non-scalar-pair layout"
        );
        let imm = Immediate::ScalarPair(a, b);
        ImmTy { imm, layout }
    }

    #[inline(always)]
    pub fn from_immediate(imm: Immediate<Prov>, layout: TyAndLayout<'tcx>) -> Self {
        // Without a `cx` we cannot call `assert_matches_abi`.
        debug_assert!(
            match (imm, layout.backend_repr) {
                (Immediate::Scalar(..), BackendRepr::Scalar(..)) => true,
                (Immediate::ScalarPair(..), BackendRepr::ScalarPair(..)) => true,
                (Immediate::Uninit, _) if layout.is_sized() => true,
                _ => false,
            },
            "immediate {imm:?} does not fit to layout {layout:?}",
        );
        ImmTy { imm, layout }
    }

    #[inline]
    pub fn uninit(layout: TyAndLayout<'tcx>) -> Self {
        debug_assert!(layout.is_sized(), "immediates must be sized");
        ImmTy { imm: Immediate::Uninit, layout }
    }

    #[inline]
    pub fn from_scalar_int(s: ScalarInt, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from(s), layout)
    }

    #[inline]
    pub fn from_uint(i: impl Into<u128>, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_uint(i, layout.size), layout)
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_int(i, layout.size), layout)
    }

    #[inline]
    pub fn from_bool(b: bool, tcx: TyCtxt<'tcx>) -> Self {
        // Can use any typing env, since `bool` is always monomorphic.
        let layout = tcx
            .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(tcx.types.bool))
            .unwrap();
        Self::from_scalar(Scalar::from_bool(b), layout)
    }

    #[inline]
    pub fn from_ordering(c: std::cmp::Ordering, tcx: TyCtxt<'tcx>) -> Self {
        // Can use any typing env, since `Ordering` is always monomorphic.
        let ty = tcx.ty_ordering_enum(DUMMY_SP);
        let layout =
            tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)).unwrap();
        Self::from_scalar(Scalar::Int(c.into()), layout)
    }

    pub fn from_pair(a: Self, b: Self, cx: &(impl HasTypingEnv<'tcx> + HasTyCtxt<'tcx>)) -> Self {
        let layout = cx
            .tcx()
            .layout_of(
                cx.typing_env().as_query_input(Ty::new_tup(cx.tcx(), &[a.layout.ty, b.layout.ty])),
            )
            .unwrap();
        Self::from_scalar_pair(a.to_scalar(), b.to_scalar(), layout)
    }

    /// Return the immediate as a `ScalarInt`. Ensures that it has the size that the layout of the
    /// immediate indicates.
    #[inline]
    pub fn to_scalar_int(&self) -> InterpResult<'tcx, ScalarInt> {
        let s = self.to_scalar().to_scalar_int()?;
        if s.size() != self.layout.size {
            throw_ub!(ScalarSizeMismatch(ScalarSizeMismatch {
                target_size: self.layout.size.bytes(),
                data_size: s.size().bytes(),
            }));
        }
        interp_ok(s)
    }

    #[inline]
    pub fn to_const_int(self) -> ConstInt {
        assert!(self.layout.ty.is_integral());
        let int = self.imm.to_scalar_int();
        assert_eq!(int.size(), self.layout.size);
        ConstInt::new(int, self.layout.ty.is_signed(), self.layout.ty.is_ptr_sized_integral())
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn to_pair(self, cx: &(impl HasTyCtxt<'tcx> + HasTypingEnv<'tcx>)) -> (Self, Self) {
        let layout = self.layout;
        let (val0, val1) = self.to_scalar_pair();
        (
            ImmTy::from_scalar(val0, layout.field(cx, 0)),
            ImmTy::from_scalar(val1, layout.field(cx, 1)),
        )
    }

    /// Compute the "sub-immediate" that is located within the `base` at the given offset with the
    /// given layout.
    // Not called `offset` to avoid confusion with the trait method.
    fn offset_(&self, offset: Size, layout: TyAndLayout<'tcx>, cx: &impl HasDataLayout) -> Self {
        // Verify that the input matches its type.
        if cfg!(debug_assertions) {
            self.assert_matches_abi(
                self.layout.backend_repr,
                "invalid input to Immediate::offset",
                cx,
            );
        }
        // `ImmTy` have already been checked to be in-bounds, so we can just check directly if this
        // remains in-bounds. This cannot actually be violated since projections are type-checked
        // and bounds-checked.
        assert!(
            offset + layout.size <= self.layout.size,
            "attempting to project to field at offset {} with size {} into immediate with layout {:#?}",
            offset.bytes(),
            layout.size.bytes(),
            self.layout,
        );
        // This makes several assumptions about what layouts we will encounter; we match what
        // codegen does as good as we can (see `extract_field` in `rustc_codegen_ssa/src/mir/operand.rs`).
        let inner_val: Immediate<_> = match (**self, self.layout.backend_repr) {
            // If the entire value is uninit, then so is the field (can happen in ConstProp).
            (Immediate::Uninit, _) => Immediate::Uninit,
            // If the field is uninhabited, we can forget the data (can happen in ConstProp).
            // `enum S { A(!), B, C }` is an example of an enum with Scalar layout that
            // has an uninhabited variant, which means this case is possible.
            _ if layout.is_uninhabited() => Immediate::Uninit,
            // the field contains no information, can be left uninit
            // (Scalar/ScalarPair can contain even aligned ZST, not just 1-ZST)
            _ if layout.is_zst() => Immediate::Uninit,
            // some fieldless enum variants can have non-zero size but still `Aggregate` ABI... try
            // to detect those here and also give them no data
            _ if matches!(layout.backend_repr, BackendRepr::Memory { .. })
                && matches!(layout.variants, abi::Variants::Single { .. })
                && matches!(&layout.fields, abi::FieldsShape::Arbitrary { offsets, .. } if offsets.len() == 0) =>
            {
                Immediate::Uninit
            }
            // the field covers the entire type
            _ if layout.size == self.layout.size => {
                assert_eq!(offset.bytes(), 0);
                **self
            }
            // extract fields from types with `ScalarPair` ABI
            (Immediate::ScalarPair(a_val, b_val), BackendRepr::ScalarPair(a, b)) => {
                Immediate::from(if offset.bytes() == 0 {
                    a_val
                } else {
                    assert_eq!(offset, a.size(cx).align_to(b.align(cx).abi));
                    b_val
                })
            }
            // everything else is a bug
            _ => bug!(
                "invalid field access on immediate {} at offset {}, original layout {:#?}",
                self,
                offset.bytes(),
                self.layout
            ),
        };
        // Ensure the new layout matches the new value.
        inner_val.assert_matches_abi(
            layout.backend_repr,
            "invalid field type in Immediate::offset",
            cx,
        );

        ImmTy::from_immediate(inner_val, layout)
    }
}

impl<'tcx, Prov: Provenance> Projectable<'tcx, Prov> for ImmTy<'tcx, Prov> {
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn meta(&self) -> MemPlaceMeta<Prov> {
        debug_assert!(self.layout.is_sized()); // unsized ImmTy can only exist temporarily and should never reach this here
        MemPlaceMeta::None
    }

    fn offset_with_meta<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        offset: Size,
        _mode: OffsetMode,
        meta: MemPlaceMeta<Prov>,
        layout: TyAndLayout<'tcx>,
        ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, Self> {
        assert_matches!(meta, MemPlaceMeta::None); // we can't store this anywhere anyway
        interp_ok(self.offset_(offset, layout, ecx))
    }

    #[inline(always)]
    fn to_op<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        _ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        interp_ok(self.clone().into())
    }
}

/// An `Operand` is the result of computing a `mir::Operand`. It can be immediate,
/// or still in memory. The latter is an optimization, to delay reading that chunk of
/// memory and to avoid having to store arbitrary-sized data here.
#[derive(Copy, Clone, Debug)]
pub(super) enum Operand<Prov: Provenance = CtfeProvenance> {
    Immediate(Immediate<Prov>),
    Indirect(MemPlace<Prov>),
}

#[derive(Clone)]
pub struct OpTy<'tcx, Prov: Provenance = CtfeProvenance> {
    op: Operand<Prov>, // Keep this private; it helps enforce invariants.
    pub layout: TyAndLayout<'tcx>,
}

impl<Prov: Provenance> std::fmt::Debug for OpTy<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Printing `layout` results in too much noise; just print a nice version of the type.
        f.debug_struct("OpTy")
            .field("op", &self.op)
            .field("ty", &format_args!("{}", self.layout.ty))
            .finish()
    }
}

impl<'tcx, Prov: Provenance> From<ImmTy<'tcx, Prov>> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn from(val: ImmTy<'tcx, Prov>) -> Self {
        OpTy { op: Operand::Immediate(val.imm), layout: val.layout }
    }
}

impl<'tcx, Prov: Provenance> From<MPlaceTy<'tcx, Prov>> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn from(mplace: MPlaceTy<'tcx, Prov>) -> Self {
        OpTy { op: Operand::Indirect(*mplace.mplace()), layout: mplace.layout }
    }
}

impl<'tcx, Prov: Provenance> OpTy<'tcx, Prov> {
    #[inline(always)]
    pub(super) fn op(&self) -> &Operand<Prov> {
        &self.op
    }
}

impl<'tcx, Prov: Provenance> Projectable<'tcx, Prov> for OpTy<'tcx, Prov> {
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline]
    fn meta(&self) -> MemPlaceMeta<Prov> {
        match self.as_mplace_or_imm() {
            Left(mplace) => mplace.meta(),
            Right(_) => {
                debug_assert!(self.layout.is_sized(), "unsized immediates are not a thing");
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
        match self.as_mplace_or_imm() {
            Left(mplace) => {
                interp_ok(mplace.offset_with_meta(offset, mode, meta, layout, ecx)?.into())
            }
            Right(imm) => {
                assert_matches!(meta, MemPlaceMeta::None); // no place to store metadata here
                // Every part of an uninit is uninit.
                interp_ok(imm.offset_(offset, layout, ecx).into())
            }
        }
    }

    #[inline(always)]
    fn to_op<M: Machine<'tcx, Provenance = Prov>>(
        &self,
        _ecx: &InterpCx<'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        interp_ok(self.clone())
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
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
            return interp_ok(None);
        }

        let Some(alloc) = self.get_place_alloc(mplace)? else {
            // zero-sized type can be left uninit
            return interp_ok(Some(ImmTy::uninit(mplace.layout)));
        };

        // It may seem like all types with `Scalar` or `ScalarPair` ABI are fair game at this point.
        // However, `MaybeUninit<u64>` is considered a `Scalar` as far as its layout is concerned --
        // and yet cannot be represented by an interpreter `Scalar`, since we have to handle the
        // case where some of the bytes are initialized and others are not. So, we need an extra
        // check that walks over the type of `mplace` to make sure it is truly correct to treat this
        // like a `Scalar` (or `ScalarPair`).
        interp_ok(match mplace.layout.backend_repr {
            BackendRepr::Scalar(abi::Scalar::Initialized { value: s, .. }) => {
                let size = s.size(self);
                assert_eq!(size, mplace.layout.size, "abi::Scalar size does not match layout size");
                let scalar = alloc.read_scalar(
                    alloc_range(Size::ZERO, size),
                    /*read_provenance*/ matches!(s, abi::Primitive::Pointer(_)),
                )?;
                Some(ImmTy::from_scalar(scalar, mplace.layout))
            }
            BackendRepr::ScalarPair(
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
                    /*read_provenance*/ matches!(a, abi::Primitive::Pointer(_)),
                )?;
                let b_val = alloc.read_scalar(
                    alloc_range(b_offset, b_size),
                    /*read_provenance*/ matches!(b, abi::Primitive::Pointer(_)),
                )?;
                Some(ImmTy::from_immediate(Immediate::ScalarPair(a_val, b_val), mplace.layout))
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
        src: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Either<MPlaceTy<'tcx, M::Provenance>, ImmTy<'tcx, M::Provenance>>> {
        interp_ok(match src.to_op(self)?.as_mplace_or_imm() {
            Left(ref mplace) => {
                if let Some(val) = self.read_immediate_from_mplace_raw(mplace)? {
                    Right(val)
                } else {
                    Left(mplace.clone())
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
        op: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        if !matches!(
            op.layout().backend_repr,
            BackendRepr::Scalar(abi::Scalar::Initialized { .. })
                | BackendRepr::ScalarPair(
                    abi::Scalar::Initialized { .. },
                    abi::Scalar::Initialized { .. }
                )
        ) {
            span_bug!(self.cur_span(), "primitive read not possible for type: {}", op.layout().ty);
        }
        let imm = self.read_immediate_raw(op)?.right().unwrap();
        if matches!(*imm, Immediate::Uninit) {
            throw_ub!(InvalidUninitBytes(None));
        }
        interp_ok(imm)
    }

    /// Read a scalar from a place
    pub fn read_scalar(
        &self,
        op: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        interp_ok(self.read_immediate(op)?.to_scalar())
    }

    // Pointer-sized reads are fairly common and need target layout access, so we wrap them in
    // convenience functions.

    /// Read a pointer from a place.
    pub fn read_pointer(
        &self,
        op: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Pointer<Option<M::Provenance>>> {
        self.read_scalar(op)?.to_pointer(self)
    }
    /// Read a pointer-sized unsigned integer from a place.
    pub fn read_target_usize(
        &self,
        op: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, u64> {
        self.read_scalar(op)?.to_target_usize(self)
    }
    /// Read a pointer-sized signed integer from a place.
    pub fn read_target_isize(
        &self,
        op: &impl Projectable<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, i64> {
        self.read_scalar(op)?.to_target_isize(self)
    }

    /// Turn the wide MPlace into a string (must already be dereferenced!)
    pub fn read_str(&self, mplace: &MPlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx, &str> {
        let len = mplace.len(self)?;
        let bytes = self.read_bytes_ptr_strip_provenance(mplace.ptr(), Size::from_bytes(len))?;
        let s = std::str::from_utf8(bytes).map_err(|err| err_ub!(InvalidStr(err)))?;
        interp_ok(s)
    }

    /// Read from a local of the current frame. Convenience method for [`InterpCx::local_at_frame_to_op`].
    pub fn local_to_op(
        &self,
        local: mir::Local,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        self.local_at_frame_to_op(self.frame(), local, layout)
    }

    /// Read from a local of a given frame.
    /// Will not access memory, instead an indirect `Operand` is returned.
    ///
    /// This is public because it is used by [Aquascope](https://github.com/cognitive-engineering-lab/aquascope/)
    /// to get an OpTy from a local.
    pub fn local_at_frame_to_op(
        &self,
        frame: &Frame<'tcx, M::Provenance, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        let layout = self.layout_of_local(frame, local, layout)?;
        let op = *frame.locals[local].access()?;
        if matches!(op, Operand::Immediate(_)) {
            assert!(!layout.is_unsized());
        }
        M::after_local_read(self, frame, local)?;
        interp_ok(OpTy { op, layout })
    }

    /// Every place can be read from, so we can turn them into an operand.
    /// This will definitely return `Indirect` if the place is a `Ptr`, i.e., this
    /// will never actually read from memory.
    pub fn place_to_op(
        &self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        match place.as_mplace_or_local() {
            Left(mplace) => interp_ok(mplace.into()),
            Right((local, offset, locals_addr, _)) => {
                debug_assert!(place.layout.is_sized()); // only sized locals can ever be `Place::Local`.
                debug_assert_eq!(locals_addr, self.frame().locals_addr());
                let base = self.local_to_op(local, None)?;
                interp_ok(match offset {
                    Some(offset) => base.offset(offset, place.layout, self)?,
                    None => {
                        // In the common case this hasn't been projected.
                        debug_assert_eq!(place.layout, base.layout);
                        base
                    }
                })
            }
        }
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

        let mut op = self.local_to_op(mir_place.local, layout)?;
        // Using `try_fold` turned out to be bad for performance, hence the loop.
        for elem in mir_place.projection.iter() {
            op = self.project(&op, elem)?
        }

        trace!("eval_place_to_op: got {:?}", op);
        // Sanity-check the type we ended up with.
        if cfg!(debug_assertions) {
            let normalized_place_ty = self
                .instantiate_from_current_frame_and_normalize_erasing_regions(
                    mir_place.ty(&self.frame().body.local_decls, *self.tcx).ty,
                )?;
            if !mir_assign_valid_types(
                *self.tcx,
                self.typing_env(),
                self.layout_of(normalized_place_ty)?,
                op.layout,
            ) {
                span_bug!(
                    self.cur_span(),
                    "eval_place of a MIR place with type {} produced an interpreter operand with type {}",
                    normalized_place_ty,
                    op.layout.ty,
                )
            }
        }
        interp_ok(op)
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
                let c = self.instantiate_from_current_frame_and_normalize_erasing_regions(
                    constant.const_,
                )?;

                // This can still fail:
                // * During ConstProp, with `TooGeneric` or since the `required_consts` were not all
                //   checked yet.
                // * During CTFE, since promoteds in `const`/`static` initializer bodies can fail.
                self.eval_mir_constant(&c, constant.span, layout)?
            }
        };
        trace!("{:?}: {:?}", mir_op, op);
        interp_ok(op)
    }

    pub(crate) fn const_val_to_op(
        &self,
        val_val: mir::ConstValue<'tcx>,
        ty: Ty<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // Other cases need layout.
        let adjust_scalar = |scalar| -> InterpResult<'tcx, _> {
            interp_ok(match scalar {
                Scalar::Ptr(ptr, size) => Scalar::Ptr(self.global_root_pointer(ptr)?, size),
                Scalar::Int(int) => Scalar::Int(int),
            })
        };
        let layout =
            from_known_layout(self.tcx, self.typing_env(), layout, || self.layout_of(ty).into())?;
        let imm = match val_val {
            mir::ConstValue::Indirect { alloc_id, offset } => {
                // This is const data, no mutation allowed.
                let ptr = self.global_root_pointer(Pointer::new(
                    CtfeProvenance::from(alloc_id).as_immutable(),
                    offset,
                ))?;
                return interp_ok(self.ptr_to_mplace(ptr.into(), layout).into());
            }
            mir::ConstValue::Scalar(x) => adjust_scalar(x)?.into(),
            mir::ConstValue::ZeroSized => Immediate::Uninit,
            mir::ConstValue::Slice { data, meta } => {
                // This is const data, no mutation allowed.
                let alloc_id = self.tcx.reserve_and_set_memory_alloc(data);
                let ptr = Pointer::new(CtfeProvenance::from(alloc_id).as_immutable(), Size::ZERO);
                Immediate::new_slice(self.global_root_pointer(ptr)?.into(), meta, self)
            }
        };
        interp_ok(OpTy { op: Operand::Immediate(imm), layout })
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(ImmTy<'_>, 64);
    static_assert_size!(Immediate, 48);
    static_assert_size!(OpTy<'_>, 72);
    static_assert_size!(Operand, 56);
    // tidy-alphabetical-end
}
