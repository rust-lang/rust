use std::ops::Bound;
use std::{cmp, fmt};

use rustc_abi::{
    AddressSpace, Align, ExternAbi, FieldIdx, FieldsShape, HasDataLayout, LayoutData, PointeeInfo,
    PointerKind, Primitive, ReprOptions, Scalar, Size, TagEncoding, TargetDataLayout,
    TyAbiInterface, VariantIdx, Variants,
};
use rustc_error_messages::DiagMessage;
use rustc_errors::{
    Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, IntoDiagArg, Level,
};
use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, extension};
use rustc_session::config::OptLevel;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, Symbol, sym};
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{HasTargetSpec, HasX86AbiOpt, PanicStrategy, Target, X86Abi};
use tracing::debug;
use {rustc_abi as abi, rustc_hir as hir};

use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::query::TyCtxtAt;
use crate::ty::normalize_erasing_regions::NormalizationError;
use crate::ty::{self, CoroutineArgsExt, Ty, TyCtxt, TypeVisitableExt};

#[extension(pub trait IntegerExt)]
impl abi::Integer {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>, signed: bool) -> Ty<'tcx> {
        use abi::Integer::{I8, I16, I32, I64, I128};
        match (*self, signed) {
            (I8, false) => tcx.types.u8,
            (I16, false) => tcx.types.u16,
            (I32, false) => tcx.types.u32,
            (I64, false) => tcx.types.u64,
            (I128, false) => tcx.types.u128,
            (I8, true) => tcx.types.i8,
            (I16, true) => tcx.types.i16,
            (I32, true) => tcx.types.i32,
            (I64, true) => tcx.types.i64,
            (I128, true) => tcx.types.i128,
        }
    }

    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: ty::IntTy) -> abi::Integer {
        use abi::Integer::{I8, I16, I32, I64, I128};
        match ity {
            ty::IntTy::I8 => I8,
            ty::IntTy::I16 => I16,
            ty::IntTy::I32 => I32,
            ty::IntTy::I64 => I64,
            ty::IntTy::I128 => I128,
            ty::IntTy::Isize => cx.data_layout().ptr_sized_integer(),
        }
    }
    fn from_uint_ty<C: HasDataLayout>(cx: &C, ity: ty::UintTy) -> abi::Integer {
        use abi::Integer::{I8, I16, I32, I64, I128};
        match ity {
            ty::UintTy::U8 => I8,
            ty::UintTy::U16 => I16,
            ty::UintTy::U32 => I32,
            ty::UintTy::U64 => I64,
            ty::UintTy::U128 => I128,
            ty::UintTy::Usize => cx.data_layout().ptr_sized_integer(),
        }
    }

    /// Finds the appropriate Integer type and signedness for the given
    /// signed discriminant range and `#[repr]` attribute.
    /// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (abi::Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = abi::Integer::fit_unsigned(cmp::max(min as u128, max as u128));
        let signed_fit = cmp::max(abi::Integer::fit_signed(min), abi::Integer::fit_signed(max));

        if let Some(ity) = repr.int {
            let discr = abi::Integer::from_attr(&tcx, ity);
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                bug!(
                    "Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum `{}`",
                    ty
                )
            }
            return (discr, ity.is_signed());
        }

        let at_least = if repr.c() {
            // This is usually I32, however it can be different on some platforms,
            // notably hexagon and arm-none/thumb-none
            tcx.data_layout().c_enum_min_size
        } else {
            // repr(Rust) enums try to be as small as possible
            abi::Integer::I8
        };

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (cmp::max(unsigned_fit, at_least), false)
        } else {
            (cmp::max(signed_fit, at_least), true)
        }
    }
}

#[extension(pub trait FloatExt)]
impl abi::Float {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        use abi::Float::*;
        match *self {
            F16 => tcx.types.f16,
            F32 => tcx.types.f32,
            F64 => tcx.types.f64,
            F128 => tcx.types.f128,
        }
    }

    fn from_float_ty(fty: ty::FloatTy) -> Self {
        use abi::Float::*;
        match fty {
            ty::FloatTy::F16 => F16,
            ty::FloatTy::F32 => F32,
            ty::FloatTy::F64 => F64,
            ty::FloatTy::F128 => F128,
        }
    }
}

#[extension(pub trait PrimitiveExt)]
impl Primitive {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            Primitive::Int(i, signed) => i.to_ty(tcx, signed),
            Primitive::Float(f) => f.to_ty(tcx),
            // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
            Primitive::Pointer(_) => Ty::new_mut_ptr(tcx, tcx.types.unit),
        }
    }

    /// Return an *integer* type matching this primitive.
    /// Useful in particular when dealing with enum discriminants.
    #[inline]
    fn to_int_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            Primitive::Int(i, signed) => i.to_ty(tcx, signed),
            // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
            Primitive::Pointer(_) => {
                let signed = false;
                tcx.data_layout().ptr_sized_integer().to_ty(tcx, signed)
            }
            Primitive::Float(_) => bug!("floats do not have an int type"),
        }
    }
}

/// The first half of a wide pointer.
///
/// - For a trait object, this is the address of the box.
/// - For a slice, this is the base address.
pub const WIDE_PTR_ADDR: usize = 0;

/// The second half of a wide pointer.
///
/// - For a trait object, this is the address of the vtable.
/// - For a slice, this is the length.
pub const WIDE_PTR_EXTRA: usize = 1;

pub const MAX_SIMD_LANES: u64 = rustc_abi::MAX_SIMD_LANES;

/// Used in `check_validity_requirement` to indicate the kind of initialization
/// that is checked to be valid
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
pub enum ValidityRequirement {
    Inhabited,
    Zero,
    /// The return value of mem::uninitialized, 0x01
    /// (unless -Zstrict-init-checks is on, in which case it's the same as Uninit).
    UninitMitigated0x01Fill,
    /// True uninitialized memory.
    Uninit,
}

impl ValidityRequirement {
    pub fn from_intrinsic(intrinsic: Symbol) -> Option<Self> {
        match intrinsic {
            sym::assert_inhabited => Some(Self::Inhabited),
            sym::assert_zero_valid => Some(Self::Zero),
            sym::assert_mem_uninitialized_valid => Some(Self::UninitMitigated0x01Fill),
            _ => None,
        }
    }
}

impl fmt::Display for ValidityRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inhabited => f.write_str("is inhabited"),
            Self::Zero => f.write_str("allows being left zeroed"),
            Self::UninitMitigated0x01Fill => f.write_str("allows being filled with 0x01"),
            Self::Uninit => f.write_str("allows being left uninitialized"),
        }
    }
}

#[derive(Copy, Clone, Debug, HashStable, TyEncodable, TyDecodable)]
pub enum LayoutError<'tcx> {
    /// A type doesn't have a sensible layout.
    ///
    /// This variant is used for layout errors that don't necessarily cause
    /// compile errors.
    ///
    /// For example, this can happen if a struct contains an unsized type in a
    /// non-tail field, but has an unsatisfiable bound like `str: Sized`.
    Unknown(Ty<'tcx>),
    /// The size of a type exceeds [`TargetDataLayout::obj_size_bound`].
    SizeOverflow(Ty<'tcx>),
    /// The layout can vary due to a generic parameter.
    ///
    /// Unlike `Unknown`, this variant is a "soft" error and indicates that the layout
    /// may become computable after further instantiating the generic parameter(s).
    TooGeneric(Ty<'tcx>),
    /// An alias failed to normalize.
    ///
    /// This variant is necessary, because, due to trait solver incompleteness, it is
    /// possible than an alias that was rigid during analysis fails to normalize after
    /// revealing opaque types.
    ///
    /// See `tests/ui/layout/normalization-failure.rs` for an example.
    NormalizationFailure(Ty<'tcx>, NormalizationError<'tcx>),
    /// A non-layout error is reported elsewhere.
    ReferencesError(ErrorGuaranteed),
    /// A type has cyclic layout, i.e. the type contains itself without indirection.
    Cycle(ErrorGuaranteed),
}

impl<'tcx> LayoutError<'tcx> {
    pub fn diagnostic_message(&self) -> DiagMessage {
        use LayoutError::*;

        use crate::fluent_generated::*;
        match self {
            Unknown(_) => middle_layout_unknown,
            SizeOverflow(_) => middle_layout_size_overflow,
            TooGeneric(_) => middle_layout_too_generic,
            NormalizationFailure(_, _) => middle_layout_normalization_failure,
            Cycle(_) => middle_layout_cycle,
            ReferencesError(_) => middle_layout_references_error,
        }
    }

    pub fn into_diagnostic(self) -> crate::error::LayoutError<'tcx> {
        use LayoutError::*;

        use crate::error::LayoutError as E;
        match self {
            Unknown(ty) => E::Unknown { ty },
            SizeOverflow(ty) => E::Overflow { ty },
            TooGeneric(ty) => E::TooGeneric { ty },
            NormalizationFailure(ty, e) => {
                E::NormalizationFailure { ty, failure_ty: e.get_type_for_failure() }
            }
            Cycle(_) => E::Cycle,
            ReferencesError(_) => E::ReferencesError,
        }
    }
}

// FIXME: Once the other errors that embed this error have been converted to translatable
// diagnostics, this Display impl should be removed.
impl<'tcx> fmt::Display for LayoutError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            LayoutError::Unknown(ty) => write!(f, "the type `{ty}` has an unknown layout"),
            LayoutError::TooGeneric(ty) => {
                write!(f, "the type `{ty}` does not have a fixed layout")
            }
            LayoutError::SizeOverflow(ty) => {
                write!(f, "values of the type `{ty}` are too big for the target architecture")
            }
            LayoutError::NormalizationFailure(t, e) => write!(
                f,
                "unable to determine layout for `{}` because `{}` cannot be normalized",
                t,
                e.get_type_for_failure()
            ),
            LayoutError::Cycle(_) => write!(f, "a cycle occurred during layout computation"),
            LayoutError::ReferencesError(_) => write!(f, "the type has an unknown layout"),
        }
    }
}

impl<'tcx> IntoDiagArg for LayoutError<'tcx> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(&mut None)
    }
}

#[derive(Clone, Copy)]
pub struct LayoutCx<'tcx> {
    pub calc: abi::LayoutCalculator<TyCtxt<'tcx>>,
    pub typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> LayoutCx<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> Self {
        Self { calc: abi::LayoutCalculator::new(tcx), typing_env }
    }
}

/// Type size "skeleton", i.e., the only information determining a type's size.
/// While this is conservative, (aside from constant sizes, only pointers,
/// newtypes thereof and null pointer optimized enums are allowed), it is
/// enough to statically check common use cases of transmute.
#[derive(Copy, Clone, Debug)]
pub enum SizeSkeleton<'tcx> {
    /// Any statically computable Layout.
    /// Alignment can be `None` if unknown.
    Known(Size, Option<Align>),

    /// This is a generic const expression (i.e. N * 2), which may contain some parameters.
    /// It must be of type usize, and represents the size of a type in bytes.
    /// It is not required to be evaluatable to a concrete value, but can be used to check
    /// that another SizeSkeleton is of equal size.
    Generic(ty::Const<'tcx>),

    /// A potentially-wide pointer.
    Pointer {
        /// If true, this pointer is never null.
        non_zero: bool,
        /// The type which determines the unsized metadata, if any,
        /// of this pointer. Either a type parameter or a projection
        /// depending on one, with regions erased.
        tail: Ty<'tcx>,
    },
}

impl<'tcx> SizeSkeleton<'tcx> {
    pub fn compute(
        ty: Ty<'tcx>,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> Result<SizeSkeleton<'tcx>, &'tcx LayoutError<'tcx>> {
        debug_assert!(!ty.has_non_region_infer());

        // First try computing a static layout.
        let err = match tcx.layout_of(typing_env.as_query_input(ty)) {
            Ok(layout) => {
                if layout.is_sized() {
                    return Ok(SizeSkeleton::Known(layout.size, Some(layout.align.abi)));
                } else {
                    // Just to be safe, don't claim a known layout for unsized types.
                    return Err(tcx.arena.alloc(LayoutError::Unknown(ty)));
                }
            }
            Err(err @ LayoutError::TooGeneric(_)) => err,
            // We can't extract SizeSkeleton info from other layout errors
            Err(
                e @ LayoutError::Cycle(_)
                | e @ LayoutError::Unknown(_)
                | e @ LayoutError::SizeOverflow(_)
                | e @ LayoutError::NormalizationFailure(..)
                | e @ LayoutError::ReferencesError(_),
            ) => return Err(e),
        };

        match *ty.kind() {
            ty::Ref(_, pointee, _) | ty::RawPtr(pointee, _) => {
                let non_zero = !ty.is_raw_ptr();

                let tail = tcx.struct_tail_raw(
                    pointee,
                    |ty| match tcx.try_normalize_erasing_regions(typing_env, ty) {
                        Ok(ty) => ty,
                        Err(e) => Ty::new_error_with_message(
                            tcx,
                            DUMMY_SP,
                            format!(
                                "normalization failed for {} but no errors reported",
                                e.get_type_for_failure()
                            ),
                        ),
                    },
                    || {},
                );

                match tail.kind() {
                    ty::Param(_) | ty::Alias(ty::Projection | ty::Inherent, _) => {
                        debug_assert!(tail.has_non_region_param());
                        Ok(SizeSkeleton::Pointer { non_zero, tail: tcx.erase_regions(tail) })
                    }
                    ty::Error(guar) => {
                        // Fixes ICE #124031
                        return Err(tcx.arena.alloc(LayoutError::ReferencesError(*guar)));
                    }
                    _ => bug!(
                        "SizeSkeleton::compute({ty}): layout errored ({err:?}), yet \
                              tail `{tail}` is not a type parameter or a projection",
                    ),
                }
            }
            ty::Array(inner, len) if tcx.features().transmute_generic_consts() => {
                let len_eval = len.try_to_target_usize(tcx);
                if len_eval == Some(0) {
                    return Ok(SizeSkeleton::Known(Size::from_bytes(0), None));
                }

                match SizeSkeleton::compute(inner, tcx, typing_env)? {
                    // This may succeed because the multiplication of two types may overflow
                    // but a single size of a nested array will not.
                    SizeSkeleton::Known(s, a) => {
                        if let Some(c) = len_eval {
                            let size = s
                                .bytes()
                                .checked_mul(c)
                                .ok_or_else(|| &*tcx.arena.alloc(LayoutError::SizeOverflow(ty)))?;
                            // Alignment is unchanged by arrays.
                            return Ok(SizeSkeleton::Known(Size::from_bytes(size), a));
                        }
                        Err(err)
                    }
                    SizeSkeleton::Pointer { .. } | SizeSkeleton::Generic(_) => Err(err),
                }
            }

            ty::Adt(def, args) => {
                // Only newtypes and enums w/ nullable pointer optimization.
                if def.is_union() || def.variants().is_empty() || def.variants().len() > 2 {
                    return Err(err);
                }

                // Get a zero-sized variant or a pointer newtype.
                let zero_or_ptr_variant = |i| {
                    let i = VariantIdx::from_usize(i);
                    let fields =
                        def.variant(i).fields.iter().map(|field| {
                            SizeSkeleton::compute(field.ty(tcx, args), tcx, typing_env)
                        });
                    let mut ptr = None;
                    for field in fields {
                        let field = field?;
                        match field {
                            SizeSkeleton::Known(size, align) => {
                                let is_1zst = size.bytes() == 0
                                    && align.is_some_and(|align| align.bytes() == 1);
                                if !is_1zst {
                                    return Err(err);
                                }
                            }
                            SizeSkeleton::Pointer { .. } => {
                                if ptr.is_some() {
                                    return Err(err);
                                }
                                ptr = Some(field);
                            }
                            SizeSkeleton::Generic(_) => {
                                return Err(err);
                            }
                        }
                    }
                    Ok(ptr)
                };

                let v0 = zero_or_ptr_variant(0)?;
                // Newtype.
                if def.variants().len() == 1 {
                    if let Some(SizeSkeleton::Pointer { non_zero, tail }) = v0 {
                        return Ok(SizeSkeleton::Pointer {
                            non_zero: non_zero
                                || match tcx.layout_scalar_valid_range(def.did()) {
                                    (Bound::Included(start), Bound::Unbounded) => start > 0,
                                    (Bound::Included(start), Bound::Included(end)) => {
                                        0 < start && start < end
                                    }
                                    _ => false,
                                },
                            tail,
                        });
                    } else {
                        return Err(err);
                    }
                }

                let v1 = zero_or_ptr_variant(1)?;
                // Nullable pointer enum optimization.
                match (v0, v1) {
                    (Some(SizeSkeleton::Pointer { non_zero: true, tail }), None)
                    | (None, Some(SizeSkeleton::Pointer { non_zero: true, tail })) => {
                        Ok(SizeSkeleton::Pointer { non_zero: false, tail })
                    }
                    _ => Err(err),
                }
            }

            ty::Alias(..) => {
                let normalized = tcx.normalize_erasing_regions(typing_env, ty);
                if ty == normalized {
                    Err(err)
                } else {
                    SizeSkeleton::compute(normalized, tcx, typing_env)
                }
            }

            // Pattern types are always the same size as their base.
            ty::Pat(base, _) => SizeSkeleton::compute(base, tcx, typing_env),

            _ => Err(err),
        }
    }

    pub fn same_size(self, other: SizeSkeleton<'tcx>) -> bool {
        match (self, other) {
            (SizeSkeleton::Known(a, _), SizeSkeleton::Known(b, _)) => a == b,
            (SizeSkeleton::Pointer { tail: a, .. }, SizeSkeleton::Pointer { tail: b, .. }) => {
                a == b
            }
            // constants are always pre-normalized into a canonical form so this
            // only needs to check if their pointers are identical.
            (SizeSkeleton::Generic(a), SizeSkeleton::Generic(b)) => a == b,
            _ => false,
        }
    }
}

pub trait HasTyCtxt<'tcx>: HasDataLayout {
    fn tcx(&self) -> TyCtxt<'tcx>;
}

pub trait HasTypingEnv<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx>;

    /// FIXME(#132279): This method should not be used as in the future
    /// everything should take a `TypingEnv` instead. Remove it as that point.
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.typing_env().param_env
    }
}

impl<'tcx> HasDataLayout for TyCtxt<'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.data_layout
    }
}

impl<'tcx> HasTargetSpec for TyCtxt<'tcx> {
    fn target_spec(&self) -> &Target {
        &self.sess.target
    }
}

impl<'tcx> HasX86AbiOpt for TyCtxt<'tcx> {
    fn x86_abi_opt(&self) -> X86Abi {
        X86Abi {
            regparm: self.sess.opts.unstable_opts.regparm,
            reg_struct_return: self.sess.opts.unstable_opts.reg_struct_return,
        }
    }
}

impl<'tcx> HasTyCtxt<'tcx> for TyCtxt<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        *self
    }
}

impl<'tcx> HasDataLayout for TyCtxtAt<'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.data_layout
    }
}

impl<'tcx> HasTargetSpec for TyCtxtAt<'tcx> {
    fn target_spec(&self) -> &Target {
        &self.sess.target
    }
}

impl<'tcx> HasTyCtxt<'tcx> for TyCtxtAt<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        **self
    }
}

impl<'tcx> HasTypingEnv<'tcx> for LayoutCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }
}

impl<'tcx> HasDataLayout for LayoutCx<'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.calc.cx.data_layout()
    }
}

impl<'tcx> HasTargetSpec for LayoutCx<'tcx> {
    fn target_spec(&self) -> &Target {
        self.calc.cx.target_spec()
    }
}

impl<'tcx> HasX86AbiOpt for LayoutCx<'tcx> {
    fn x86_abi_opt(&self) -> X86Abi {
        self.calc.cx.x86_abi_opt()
    }
}

impl<'tcx> HasTyCtxt<'tcx> for LayoutCx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.calc.cx
    }
}

pub trait MaybeResult<T> {
    type Error;

    fn from(x: Result<T, Self::Error>) -> Self;
    fn to_result(self) -> Result<T, Self::Error>;
}

impl<T> MaybeResult<T> for T {
    type Error = !;

    fn from(Ok(x): Result<T, Self::Error>) -> Self {
        x
    }
    fn to_result(self) -> Result<T, Self::Error> {
        Ok(self)
    }
}

impl<T, E> MaybeResult<T> for Result<T, E> {
    type Error = E;

    fn from(x: Result<T, Self::Error>) -> Self {
        x
    }
    fn to_result(self) -> Result<T, Self::Error> {
        self
    }
}

pub type TyAndLayout<'tcx> = rustc_abi::TyAndLayout<'tcx, Ty<'tcx>>;

/// Trait for contexts that want to be able to compute layouts of types.
/// This automatically gives access to `LayoutOf`, through a blanket `impl`.
pub trait LayoutOfHelpers<'tcx>: HasDataLayout + HasTyCtxt<'tcx> + HasTypingEnv<'tcx> {
    /// The `TyAndLayout`-wrapping type (or `TyAndLayout` itself), which will be
    /// returned from `layout_of` (see also `handle_layout_err`).
    type LayoutOfResult: MaybeResult<TyAndLayout<'tcx>> = TyAndLayout<'tcx>;

    /// `Span` to use for `tcx.at(span)`, from `layout_of`.
    // FIXME(eddyb) perhaps make this mandatory to get contexts to track it better?
    #[inline]
    fn layout_tcx_at_span(&self) -> Span {
        DUMMY_SP
    }

    /// Helper used for `layout_of`, to adapt `tcx.layout_of(...)` into a
    /// `Self::LayoutOfResult` (which does not need to be a `Result<...>`).
    ///
    /// Most `impl`s, which propagate `LayoutError`s, should simply return `err`,
    /// but this hook allows e.g. codegen to return only `TyAndLayout` from its
    /// `cx.layout_of(...)`, without any `Result<...>` around it to deal with
    /// (and any `LayoutError`s are turned into fatal errors or ICEs).
    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        span: Span,
        ty: Ty<'tcx>,
    ) -> <Self::LayoutOfResult as MaybeResult<TyAndLayout<'tcx>>>::Error;
}

/// Blanket extension trait for contexts that can compute layouts of types.
pub trait LayoutOf<'tcx>: LayoutOfHelpers<'tcx> {
    /// Computes the layout of a type. Note that this implicitly
    /// executes in `TypingMode::PostAnalysis`, and will normalize the input type.
    #[inline]
    fn layout_of(&self, ty: Ty<'tcx>) -> Self::LayoutOfResult {
        self.spanned_layout_of(ty, DUMMY_SP)
    }

    /// Computes the layout of a type, at `span`. Note that this implicitly
    /// executes in `TypingMode::PostAnalysis`, and will normalize the input type.
    // FIXME(eddyb) avoid passing information like this, and instead add more
    // `TyCtxt::at`-like APIs to be able to do e.g. `cx.at(span).layout_of(ty)`.
    #[inline]
    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::LayoutOfResult {
        let span = if !span.is_dummy() { span } else { self.layout_tcx_at_span() };
        let tcx = self.tcx().at(span);

        MaybeResult::from(
            tcx.layout_of(self.typing_env().as_query_input(ty))
                .map_err(|err| self.handle_layout_err(*err, span, ty)),
        )
    }
}

impl<'tcx, C: LayoutOfHelpers<'tcx>> LayoutOf<'tcx> for C {}

impl<'tcx> LayoutOfHelpers<'tcx> for LayoutCx<'tcx> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, &'tcx LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        _: Span,
        _: Ty<'tcx>,
    ) -> &'tcx LayoutError<'tcx> {
        self.tcx().arena.alloc(err)
    }
}

impl<'tcx, C> TyAbiInterface<'tcx, C> for Ty<'tcx>
where
    C: HasTyCtxt<'tcx> + HasTypingEnv<'tcx>,
{
    fn ty_and_layout_for_variant(
        this: TyAndLayout<'tcx>,
        cx: &C,
        variant_index: VariantIdx,
    ) -> TyAndLayout<'tcx> {
        let layout = match this.variants {
            // If all variants but one are uninhabited, the variant layout is the enum layout.
            Variants::Single { index } if index == variant_index => {
                return this;
            }

            Variants::Single { .. } | Variants::Empty => {
                // Single-variant and no-variant enums *can* have other variants, but those are
                // uninhabited. Produce a layout that has the right fields for that variant, so that
                // the rest of the compiler can project fields etc as usual.

                let tcx = cx.tcx();
                let typing_env = cx.typing_env();

                // Deny calling for_variant more than once for non-Single enums.
                if let Ok(original_layout) = tcx.layout_of(typing_env.as_query_input(this.ty)) {
                    assert_eq!(original_layout.variants, this.variants);
                }

                let fields = match this.ty.kind() {
                    ty::Adt(def, _) if def.variants().is_empty() => {
                        bug!("for_variant called on zero-variant enum {}", this.ty)
                    }
                    ty::Adt(def, _) => def.variant(variant_index).fields.len(),
                    _ => bug!("`ty_and_layout_for_variant` on unexpected type {}", this.ty),
                };
                tcx.mk_layout(LayoutData::uninhabited_variant(cx, variant_index, fields))
            }

            Variants::Multiple { ref variants, .. } => {
                cx.tcx().mk_layout(variants[variant_index].clone())
            }
        };

        assert_eq!(*layout.variants(), Variants::Single { index: variant_index });

        TyAndLayout { ty: this.ty, layout }
    }

    fn ty_and_layout_field(this: TyAndLayout<'tcx>, cx: &C, i: usize) -> TyAndLayout<'tcx> {
        enum TyMaybeWithLayout<'tcx> {
            Ty(Ty<'tcx>),
            TyAndLayout(TyAndLayout<'tcx>),
        }

        fn field_ty_or_layout<'tcx>(
            this: TyAndLayout<'tcx>,
            cx: &(impl HasTyCtxt<'tcx> + HasTypingEnv<'tcx>),
            i: usize,
        ) -> TyMaybeWithLayout<'tcx> {
            let tcx = cx.tcx();
            let tag_layout = |tag: Scalar| -> TyAndLayout<'tcx> {
                TyAndLayout {
                    layout: tcx.mk_layout(LayoutData::scalar(cx, tag)),
                    ty: tag.primitive().to_ty(tcx),
                }
            };

            match *this.ty.kind() {
                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::FnPtr(..)
                | ty::Never
                | ty::FnDef(..)
                | ty::CoroutineWitness(..)
                | ty::Foreign(..)
                | ty::Pat(_, _)
                | ty::Dynamic(_, _, ty::Dyn) => {
                    bug!("TyAndLayout::field({:?}): not applicable", this)
                }

                ty::UnsafeBinder(bound_ty) => {
                    let ty = tcx.instantiate_bound_regions_with_erased(bound_ty.into());
                    field_ty_or_layout(TyAndLayout { ty, ..this }, cx, i)
                }

                // Potentially-wide pointers.
                ty::Ref(_, pointee, _) | ty::RawPtr(pointee, _) => {
                    assert!(i < this.fields.count());

                    // Reuse the wide `*T` type as its own thin pointer data field.
                    // This provides information about, e.g., DST struct pointees
                    // (which may have no non-DST form), and will work as long
                    // as the `Abi` or `FieldsShape` is checked by users.
                    if i == 0 {
                        let nil = tcx.types.unit;
                        let unit_ptr_ty = if this.ty.is_raw_ptr() {
                            Ty::new_mut_ptr(tcx, nil)
                        } else {
                            Ty::new_mut_ref(tcx, tcx.lifetimes.re_static, nil)
                        };

                        // NOTE: using an fully monomorphized typing env and `unwrap`-ing
                        // the `Result` should always work because the type is always either
                        // `*mut ()` or `&'static mut ()`.
                        let typing_env = ty::TypingEnv::fully_monomorphized();
                        return TyMaybeWithLayout::TyAndLayout(TyAndLayout {
                            ty: this.ty,
                            ..tcx.layout_of(typing_env.as_query_input(unit_ptr_ty)).unwrap()
                        });
                    }

                    let mk_dyn_vtable = |principal: Option<ty::PolyExistentialTraitRef<'tcx>>| {
                        let min_count = ty::vtable_min_entries(
                            tcx,
                            principal.map(|principal| {
                                tcx.instantiate_bound_regions_with_erased(principal)
                            }),
                        );
                        Ty::new_imm_ref(
                            tcx,
                            tcx.lifetimes.re_static,
                            // FIXME: properly type (e.g. usize and fn pointers) the fields.
                            Ty::new_array(tcx, tcx.types.usize, min_count.try_into().unwrap()),
                        )
                    };

                    let metadata = if let Some(metadata_def_id) = tcx.lang_items().metadata_type()
                        // Projection eagerly bails out when the pointee references errors,
                        // fall back to structurally deducing metadata.
                        && !pointee.references_error()
                    {
                        let metadata = tcx.normalize_erasing_regions(
                            cx.typing_env(),
                            Ty::new_projection(tcx, metadata_def_id, [pointee]),
                        );

                        // Map `Metadata = DynMetadata<dyn Trait>` back to a vtable, since it
                        // offers better information than `std::ptr::metadata::VTable`,
                        // and we rely on this layout information to trigger a panic in
                        // `std::mem::uninitialized::<&dyn Trait>()`, for example.
                        if let ty::Adt(def, args) = metadata.kind()
                            && tcx.is_lang_item(def.did(), LangItem::DynMetadata)
                            && let ty::Dynamic(data, _, ty::Dyn) = args.type_at(0).kind()
                        {
                            mk_dyn_vtable(data.principal())
                        } else {
                            metadata
                        }
                    } else {
                        match tcx.struct_tail_for_codegen(pointee, cx.typing_env()).kind() {
                            ty::Slice(_) | ty::Str => tcx.types.usize,
                            ty::Dynamic(data, _, ty::Dyn) => mk_dyn_vtable(data.principal()),
                            _ => bug!("TyAndLayout::field({:?}): not applicable", this),
                        }
                    };

                    TyMaybeWithLayout::Ty(metadata)
                }

                // Arrays and slices.
                ty::Array(element, _) | ty::Slice(element) => TyMaybeWithLayout::Ty(element),
                ty::Str => TyMaybeWithLayout::Ty(tcx.types.u8),

                // Tuples, coroutines and closures.
                ty::Closure(_, args) => field_ty_or_layout(
                    TyAndLayout { ty: args.as_closure().tupled_upvars_ty(), ..this },
                    cx,
                    i,
                ),

                ty::CoroutineClosure(_, args) => field_ty_or_layout(
                    TyAndLayout { ty: args.as_coroutine_closure().tupled_upvars_ty(), ..this },
                    cx,
                    i,
                ),

                ty::Coroutine(def_id, args) => match this.variants {
                    Variants::Empty => unreachable!(),
                    Variants::Single { index } => TyMaybeWithLayout::Ty(
                        args.as_coroutine()
                            .state_tys(def_id, tcx)
                            .nth(index.as_usize())
                            .unwrap()
                            .nth(i)
                            .unwrap(),
                    ),
                    Variants::Multiple { tag, tag_field, .. } => {
                        if FieldIdx::from_usize(i) == tag_field {
                            return TyMaybeWithLayout::TyAndLayout(tag_layout(tag));
                        }
                        TyMaybeWithLayout::Ty(args.as_coroutine().prefix_tys()[i])
                    }
                },

                ty::Tuple(tys) => TyMaybeWithLayout::Ty(tys[i]),

                // ADTs.
                ty::Adt(def, args) => {
                    match this.variants {
                        Variants::Single { index } => {
                            let field = &def.variant(index).fields[FieldIdx::from_usize(i)];
                            TyMaybeWithLayout::Ty(field.ty(tcx, args))
                        }
                        Variants::Empty => panic!("there is no field in Variants::Empty types"),

                        // Discriminant field for enums (where applicable).
                        Variants::Multiple { tag, .. } => {
                            assert_eq!(i, 0);
                            return TyMaybeWithLayout::TyAndLayout(tag_layout(tag));
                        }
                    }
                }

                ty::Alias(..)
                | ty::Bound(..)
                | ty::Placeholder(..)
                | ty::Param(_)
                | ty::Infer(_)
                | ty::Error(_) => bug!("TyAndLayout::field: unexpected type `{}`", this.ty),
            }
        }

        match field_ty_or_layout(this, cx, i) {
            TyMaybeWithLayout::Ty(field_ty) => {
                cx.tcx().layout_of(cx.typing_env().as_query_input(field_ty)).unwrap_or_else(|e| {
                    bug!(
                        "failed to get layout for `{field_ty}`: {e:?},\n\
                         despite it being a field (#{i}) of an existing layout: {this:#?}",
                    )
                })
            }
            TyMaybeWithLayout::TyAndLayout(field_layout) => field_layout,
        }
    }

    /// Compute the information for the pointer stored at the given offset inside this type.
    /// This will recurse into fields of ADTs to find the inner pointer.
    fn ty_and_layout_pointee_info_at(
        this: TyAndLayout<'tcx>,
        cx: &C,
        offset: Size,
    ) -> Option<PointeeInfo> {
        let tcx = cx.tcx();
        let typing_env = cx.typing_env();

        let pointee_info = match *this.ty.kind() {
            ty::RawPtr(p_ty, _) if offset.bytes() == 0 => {
                tcx.layout_of(typing_env.as_query_input(p_ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: None,
                })
            }
            ty::FnPtr(..) if offset.bytes() == 0 => {
                tcx.layout_of(typing_env.as_query_input(this.ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: None,
                })
            }
            ty::Ref(_, ty, mt) if offset.bytes() == 0 => {
                // Use conservative pointer kind if not optimizing. This saves us the
                // Freeze/Unpin queries, and can save time in the codegen backend (noalias
                // attributes in LLVM have compile-time cost even in unoptimized builds).
                let optimize = tcx.sess.opts.optimize != OptLevel::No;
                let kind = match mt {
                    hir::Mutability::Not => {
                        PointerKind::SharedRef { frozen: optimize && ty.is_freeze(tcx, typing_env) }
                    }
                    hir::Mutability::Mut => {
                        PointerKind::MutableRef { unpin: optimize && ty.is_unpin(tcx, typing_env) }
                    }
                };

                tcx.layout_of(typing_env.as_query_input(ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: Some(kind),
                })
            }

            _ => {
                let mut data_variant = match &this.variants {
                    // Within the discriminant field, only the niche itself is
                    // always initialized, so we only check for a pointer at its
                    // offset.
                    //
                    // Our goal here is to check whether this represents a
                    // "dereferenceable or null" pointer, so we need to ensure
                    // that there is only one other variant, and it must be null.
                    // Below, we will then check whether the pointer is indeed
                    // dereferenceable.
                    Variants::Multiple {
                        tag_encoding:
                            TagEncoding::Niche { untagged_variant, niche_variants, niche_start },
                        tag_field,
                        variants,
                        ..
                    } if variants.len() == 2
                        && this.fields.offset(tag_field.as_usize()) == offset =>
                    {
                        let tagged_variant = if *untagged_variant == VariantIdx::ZERO {
                            VariantIdx::from_u32(1)
                        } else {
                            VariantIdx::from_u32(0)
                        };
                        assert_eq!(tagged_variant, *niche_variants.start());
                        if *niche_start == 0 {
                            // The other variant is encoded as "null", so we can recurse searching for
                            // a pointer here. This relies on the fact that the codegen backend
                            // only adds "dereferenceable" if there's also a "nonnull" proof,
                            // and that null is aligned for all alignments so it's okay to forward
                            // the pointer's alignment.
                            Some(this.for_variant(cx, *untagged_variant))
                        } else {
                            None
                        }
                    }
                    Variants::Multiple { .. } => None,
                    _ => Some(this),
                };

                if let Some(variant) = data_variant {
                    // We're not interested in any unions.
                    if let FieldsShape::Union(_) = variant.fields {
                        data_variant = None;
                    }
                }

                let mut result = None;

                if let Some(variant) = data_variant {
                    // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
                    // (requires passing in the expected address space from the caller)
                    let ptr_end = offset + Primitive::Pointer(AddressSpace::DATA).size(cx);
                    for i in 0..variant.fields.count() {
                        let field_start = variant.fields.offset(i);
                        if field_start <= offset {
                            let field = variant.field(cx, i);
                            result = field.to_result().ok().and_then(|field| {
                                if ptr_end <= field_start + field.size {
                                    // We found the right field, look inside it.
                                    let field_info =
                                        field.pointee_info_at(cx, offset - field_start);
                                    field_info
                                } else {
                                    None
                                }
                            });
                            if result.is_some() {
                                break;
                            }
                        }
                    }
                }

                // Fixup info for the first field of a `Box`. Recursive traversal will have found
                // the raw pointer, so size and align are set to the boxed type, but `pointee.safe`
                // will still be `None`.
                if let Some(ref mut pointee) = result {
                    if offset.bytes() == 0
                        && let Some(boxed_ty) = this.ty.boxed_ty()
                    {
                        debug_assert!(pointee.safe.is_none());
                        let optimize = tcx.sess.opts.optimize != OptLevel::No;
                        pointee.safe = Some(PointerKind::Box {
                            unpin: optimize && boxed_ty.is_unpin(tcx, typing_env),
                            global: this.ty.is_box_global(tcx),
                        });
                    }
                }

                result
            }
        };

        debug!(
            "pointee_info_at (offset={:?}, type kind: {:?}) => {:?}",
            offset,
            this.ty.kind(),
            pointee_info
        );

        pointee_info
    }

    fn is_adt(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Adt(..))
    }

    fn is_never(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Never)
    }

    fn is_tuple(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Tuple(..))
    }

    fn is_unit(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Tuple(list) if list.len() == 0)
    }

    fn is_transparent(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Adt(def, _) if def.repr().transparent())
    }
}

/// Calculates whether a function's ABI can unwind or not.
///
/// This takes two primary parameters:
///
/// * `fn_def_id` - the `DefId` of the function. If this is provided then we can
///   determine more precisely if the function can unwind. If this is not provided
///   then we will only infer whether the function can unwind or not based on the
///   ABI of the function. For example, a function marked with `#[rustc_nounwind]`
///   is known to not unwind even if it's using Rust ABI.
///
/// * `abi` - this is the ABI that the function is defined with. This is the
///   primary factor for determining whether a function can unwind or not.
///
/// Note that in this case unwinding is not necessarily panicking in Rust. Rust
/// panics are implemented with unwinds on most platform (when
/// `-Cpanic=unwind`), but this also accounts for `-Cpanic=abort` build modes.
/// Notably unwinding is disallowed for more non-Rust ABIs unless it's
/// specifically in the name (e.g. `"C-unwind"`). Unwinding within each ABI is
/// defined for each ABI individually, but it always corresponds to some form of
/// stack-based unwinding (the exact mechanism of which varies
/// platform-by-platform).
///
/// Rust functions are classified whether or not they can unwind based on the
/// active "panic strategy". In other words Rust functions are considered to
/// unwind in `-Cpanic=unwind` mode and cannot unwind in `-Cpanic=abort` mode.
/// Note that Rust supports intermingling panic=abort and panic=unwind code, but
/// only if the final panic mode is panic=abort. In this scenario any code
/// previously compiled assuming that a function can unwind is still correct, it
/// just never happens to actually unwind at runtime.
///
/// This function's answer to whether or not a function can unwind is quite
/// impactful throughout the compiler. This affects things like:
///
/// * Calling a function which can't unwind means codegen simply ignores any
///   associated unwinding cleanup.
/// * Calling a function which can unwind from a function which can't unwind
///   causes the `abort_unwinding_calls` MIR pass to insert a landing pad that
///   aborts the process.
/// * This affects whether functions have the LLVM `nounwind` attribute, which
///   affects various optimizations and codegen.
#[inline]
#[tracing::instrument(level = "debug", skip(tcx))]
pub fn fn_can_unwind(tcx: TyCtxt<'_>, fn_def_id: Option<DefId>, abi: ExternAbi) -> bool {
    if let Some(did) = fn_def_id {
        // Special attribute for functions which can't unwind.
        if tcx.codegen_fn_attrs(did).flags.contains(CodegenFnAttrFlags::NEVER_UNWIND) {
            return false;
        }

        // With `-C panic=abort`, all non-FFI functions are required to not unwind.
        //
        // Note that this is true regardless ABI specified on the function -- a `extern "C-unwind"`
        // function defined in Rust is also required to abort.
        if tcx.sess.panic_strategy() == PanicStrategy::Abort && !tcx.is_foreign_item(did) {
            return false;
        }

        // With -Z panic-in-drop=abort, drop_in_place never unwinds.
        //
        // This is not part of `codegen_fn_attrs` as it can differ between crates
        // and therefore cannot be computed in core.
        if tcx.sess.opts.unstable_opts.panic_in_drop == PanicStrategy::Abort
            && tcx.is_lang_item(did, LangItem::DropInPlace)
        {
            return false;
        }
    }

    // Otherwise if this isn't special then unwinding is generally determined by
    // the ABI of the itself. ABIs like `C` have variants which also
    // specifically allow unwinding (`C-unwind`), but not all platform-specific
    // ABIs have such an option. Otherwise the only other thing here is Rust
    // itself, and those ABIs are determined by the panic strategy configured
    // for this compilation.
    use ExternAbi::*;
    match abi {
        C { unwind }
        | System { unwind }
        | Cdecl { unwind }
        | Stdcall { unwind }
        | Fastcall { unwind }
        | Vectorcall { unwind }
        | Thiscall { unwind }
        | Aapcs { unwind }
        | Win64 { unwind }
        | SysV64 { unwind } => unwind,
        PtxKernel
        | Msp430Interrupt
        | X86Interrupt
        | GpuKernel
        | EfiApi
        | AvrInterrupt
        | AvrNonBlockingInterrupt
        | CmseNonSecureCall
        | CmseNonSecureEntry
        | Custom
        | RiscvInterruptM
        | RiscvInterruptS
        | RustInvalid
        | Unadjusted => false,
        Rust | RustCall | RustCold => tcx.sess.panic_strategy() == PanicStrategy::Unwind,
    }
}

/// Error produced by attempting to compute or adjust a `FnAbi`.
#[derive(Copy, Clone, Debug, HashStable)]
pub enum FnAbiError<'tcx> {
    /// Error produced by a `layout_of` call, while computing `FnAbi` initially.
    Layout(LayoutError<'tcx>),
}

impl<'a, 'b, G: EmissionGuarantee> Diagnostic<'a, G> for FnAbiError<'b> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        match self {
            Self::Layout(e) => e.into_diagnostic().into_diag(dcx, level),
        }
    }
}

// FIXME(eddyb) maybe use something like this for an unified `fn_abi_of`, not
// just for error handling.
#[derive(Debug)]
pub enum FnAbiRequest<'tcx> {
    OfFnPtr { sig: ty::PolyFnSig<'tcx>, extra_args: &'tcx ty::List<Ty<'tcx>> },
    OfInstance { instance: ty::Instance<'tcx>, extra_args: &'tcx ty::List<Ty<'tcx>> },
}

/// Trait for contexts that want to be able to compute `FnAbi`s.
/// This automatically gives access to `FnAbiOf`, through a blanket `impl`.
pub trait FnAbiOfHelpers<'tcx>: LayoutOfHelpers<'tcx> {
    /// The `&FnAbi`-wrapping type (or `&FnAbi` itself), which will be
    /// returned from `fn_abi_of_*` (see also `handle_fn_abi_err`).
    type FnAbiOfResult: MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>> = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    /// Helper used for `fn_abi_of_*`, to adapt `tcx.fn_abi_of_*(...)` into a
    /// `Self::FnAbiOfResult` (which does not need to be a `Result<...>`).
    ///
    /// Most `impl`s, which propagate `FnAbiError`s, should simply return `err`,
    /// but this hook allows e.g. codegen to return only `&FnAbi` from its
    /// `cx.fn_abi_of_*(...)`, without any `Result<...>` around it to deal with
    /// (and any `FnAbiError`s are turned into fatal errors or ICEs).
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>>>::Error;
}

/// Blanket extension trait for contexts that can compute `FnAbi`s.
pub trait FnAbiOf<'tcx>: FnAbiOfHelpers<'tcx> {
    /// Compute a `FnAbi` suitable for indirect calls, i.e. to `fn` pointers.
    ///
    /// NB: this doesn't handle virtual calls - those should use `fn_abi_of_instance`
    /// instead, where the instance is an `InstanceKind::Virtual`.
    #[inline]
    fn fn_abi_of_fn_ptr(
        &self,
        sig: ty::PolyFnSig<'tcx>,
        extra_args: &'tcx ty::List<Ty<'tcx>>,
    ) -> Self::FnAbiOfResult {
        // FIXME(eddyb) get a better `span` here.
        let span = self.layout_tcx_at_span();
        let tcx = self.tcx().at(span);

        MaybeResult::from(
            tcx.fn_abi_of_fn_ptr(self.typing_env().as_query_input((sig, extra_args))).map_err(
                |err| self.handle_fn_abi_err(*err, span, FnAbiRequest::OfFnPtr { sig, extra_args }),
            ),
        )
    }

    /// Compute a `FnAbi` suitable for declaring/defining an `fn` instance, and for
    /// direct calls to an `fn`.
    ///
    /// NB: that includes virtual calls, which are represented by "direct calls"
    /// to an `InstanceKind::Virtual` instance (of `<dyn Trait as Trait>::fn`).
    #[inline]
    #[tracing::instrument(level = "debug", skip(self))]
    fn fn_abi_of_instance(
        &self,
        instance: ty::Instance<'tcx>,
        extra_args: &'tcx ty::List<Ty<'tcx>>,
    ) -> Self::FnAbiOfResult {
        // FIXME(eddyb) get a better `span` here.
        let span = self.layout_tcx_at_span();
        let tcx = self.tcx().at(span);

        MaybeResult::from(
            tcx.fn_abi_of_instance(self.typing_env().as_query_input((instance, extra_args)))
                .map_err(|err| {
                    // HACK(eddyb) at least for definitions of/calls to `Instance`s,
                    // we can get some kind of span even if one wasn't provided.
                    // However, we don't do this early in order to avoid calling
                    // `def_span` unconditionally (which may have a perf penalty).
                    let span =
                        if !span.is_dummy() { span } else { tcx.def_span(instance.def_id()) };
                    self.handle_fn_abi_err(
                        *err,
                        span,
                        FnAbiRequest::OfInstance { instance, extra_args },
                    )
                }),
        )
    }
}

impl<'tcx, C: FnAbiOfHelpers<'tcx>> FnAbiOf<'tcx> for C {}

impl<'tcx> TyCtxt<'tcx> {
    pub fn offset_of_subfield<I>(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        mut layout: TyAndLayout<'tcx>,
        indices: I,
    ) -> Size
    where
        I: Iterator<Item = (VariantIdx, FieldIdx)>,
    {
        let cx = LayoutCx::new(self, typing_env);
        let mut offset = Size::ZERO;

        for (variant, field) in indices {
            layout = layout.for_variant(&cx, variant);
            let index = field.index();
            offset += layout.fields.offset(index);
            layout = layout.field(&cx, index);
            if !layout.is_sized() {
                // If it is not sized, then the tail must still have at least a known static alignment.
                let tail = self.struct_tail_for_codegen(layout.ty, typing_env);
                if !matches!(tail.kind(), ty::Slice(..)) {
                    bug!(
                        "offset of not-statically-aligned field (type {:?}) cannot be computed statically",
                        layout.ty
                    );
                }
            }
        }

        offset
    }
}
