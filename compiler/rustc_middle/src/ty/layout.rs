use crate::error::UnsupportedFnAbi;
use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::query::TyCtxtAt;
use crate::ty::normalize_erasing_regions::NormalizationError;
use crate::ty::{self, ConstKind, ReprOptions, Ty, TyCtxt, TypeVisitableExt};
use rustc_error_messages::DiagnosticMessage;
use rustc_errors::{DiagnosticBuilder, Handler, IntoDiagnostic};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_session::config::OptLevel;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::*;
use rustc_target::spec::{abi::Abi as SpecAbi, HasTargetSpec, PanicStrategy, Target};

use std::cmp;
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Bound;

pub trait IntegerExt {
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>, signed: bool) -> Ty<'tcx>;
    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: ty::IntTy) -> Integer;
    fn from_uint_ty<C: HasDataLayout>(cx: &C, uty: ty::UintTy) -> Integer;
    fn repr_discr<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (Integer, bool);
}

impl IntegerExt for Integer {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>, signed: bool) -> Ty<'tcx> {
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

    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: ty::IntTy) -> Integer {
        match ity {
            ty::IntTy::I8 => I8,
            ty::IntTy::I16 => I16,
            ty::IntTy::I32 => I32,
            ty::IntTy::I64 => I64,
            ty::IntTy::I128 => I128,
            ty::IntTy::Isize => cx.data_layout().ptr_sized_integer(),
        }
    }
    fn from_uint_ty<C: HasDataLayout>(cx: &C, ity: ty::UintTy) -> Integer {
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
    ) -> (Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u128, max as u128));
        let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(&tcx, ity);
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
            I8
        };

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (cmp::max(unsigned_fit, at_least), false)
        } else {
            (cmp::max(signed_fit, at_least), true)
        }
    }
}

pub trait PrimitiveExt {
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
    fn to_int_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
}

impl PrimitiveExt for Primitive {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            Int(i, signed) => i.to_ty(tcx, signed),
            F32 => tcx.types.f32,
            F64 => tcx.types.f64,
            // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
            Pointer(_) => Ty::new_mut_ptr(tcx, Ty::new_unit(tcx)),
        }
    }

    /// Return an *integer* type matching this primitive.
    /// Useful in particular when dealing with enum discriminants.
    #[inline]
    fn to_int_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            Int(i, signed) => i.to_ty(tcx, signed),
            // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
            Pointer(_) => {
                let signed = false;
                tcx.data_layout().ptr_sized_integer().to_ty(tcx, signed)
            }
            F32 | F64 => bug!("floats do not have an int type"),
        }
    }
}

/// The first half of a fat pointer.
///
/// - For a trait object, this is the address of the box.
/// - For a slice, this is the base address.
pub const FAT_PTR_ADDR: usize = 0;

/// The second half of a fat pointer.
///
/// - For a trait object, this is the address of the vtable.
/// - For a slice, this is the length.
pub const FAT_PTR_EXTRA: usize = 1;

/// The maximum supported number of lanes in a SIMD vector.
///
/// This value is selected based on backend support:
/// * LLVM does not appear to have a vector width limit.
/// * Cranelift stores the base-2 log of the lane count in a 4 bit integer.
pub const MAX_SIMD_LANES: u64 = 1 << 0xF;

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
    Unknown(Ty<'tcx>),
    SizeOverflow(Ty<'tcx>),
    NormalizationFailure(Ty<'tcx>, NormalizationError<'tcx>),
    Cycle,
}

impl<'tcx> LayoutError<'tcx> {
    pub fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use LayoutError::*;
        match self {
            Unknown(_) => middle_unknown_layout,
            SizeOverflow(_) => middle_values_too_big,
            NormalizationFailure(_, _) => middle_cannot_be_normalized,
            Cycle => middle_cycle,
        }
    }

    pub fn into_diagnostic(self) -> crate::error::LayoutError<'tcx> {
        use crate::error::LayoutError as E;
        use LayoutError::*;
        match self {
            Unknown(ty) => E::Unknown { ty },
            SizeOverflow(ty) => E::Overflow { ty },
            NormalizationFailure(ty, e) => {
                E::NormalizationFailure { ty, failure_ty: e.get_type_for_failure() }
            }
            Cycle => E::Cycle,
        }
    }
}

// FIXME: Once the other errors that embed this error have been converted to translatable
// diagnostics, this Display impl should be removed.
impl<'tcx> fmt::Display for LayoutError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            LayoutError::Unknown(ty) => write!(f, "the type `{}` has an unknown layout", ty),
            LayoutError::SizeOverflow(ty) => {
                write!(f, "values of the type `{}` are too big for the current architecture", ty)
            }
            LayoutError::NormalizationFailure(t, e) => write!(
                f,
                "unable to determine layout for `{}` because `{}` cannot be normalized",
                t,
                e.get_type_for_failure()
            ),
            LayoutError::Cycle => write!(f, "a cycle occurred during layout computation"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct LayoutCx<'tcx, C> {
    pub tcx: C,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> LayoutCalculator for LayoutCx<'tcx, TyCtxt<'tcx>> {
    type TargetDataLayoutRef = &'tcx TargetDataLayout;

    fn delay_bug(&self, txt: String) {
        self.tcx.sess.delay_span_bug(DUMMY_SP, txt);
    }

    fn current_data_layout(&self) -> Self::TargetDataLayoutRef {
        &self.tcx.data_layout
    }
}

/// Type size "skeleton", i.e., the only information determining a type's size.
/// While this is conservative, (aside from constant sizes, only pointers,
/// newtypes thereof and null pointer optimized enums are allowed), it is
/// enough to statically check common use cases of transmute.
#[derive(Copy, Clone, Debug)]
pub enum SizeSkeleton<'tcx> {
    /// Any statically computable Layout.
    Known(Size),

    /// This is a generic const expression (i.e. N * 2), which may contain some parameters.
    /// It must be of type usize, and represents the size of a type in bytes.
    /// It is not required to be evaluatable to a concrete value, but can be used to check
    /// that another SizeSkeleton is of equal size.
    Generic(ty::Const<'tcx>),

    /// A potentially-fat pointer.
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
        param_env: ty::ParamEnv<'tcx>,
    ) -> Result<SizeSkeleton<'tcx>, &'tcx LayoutError<'tcx>> {
        debug_assert!(!ty.has_non_region_infer());

        // First try computing a static layout.
        let err = match tcx.layout_of(param_env.and(ty)) {
            Ok(layout) => {
                return Ok(SizeSkeleton::Known(layout.size));
            }
            Err(err @ LayoutError::Unknown(_)) => err,
            // We can't extract SizeSkeleton info from other layout errors
            Err(
                e @ LayoutError::Cycle
                | e @ LayoutError::SizeOverflow(_)
                | e @ LayoutError::NormalizationFailure(..),
            ) => return Err(e),
        };

        match *ty.kind() {
            ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                let non_zero = !ty.is_unsafe_ptr();
                let tail = tcx.struct_tail_erasing_lifetimes(pointee, param_env);
                match tail.kind() {
                    ty::Param(_) | ty::Alias(ty::Projection | ty::Inherent, _) => {
                        debug_assert!(tail.has_non_region_param());
                        Ok(SizeSkeleton::Pointer { non_zero, tail: tcx.erase_regions(tail) })
                    }
                    _ => bug!(
                        "SizeSkeleton::compute({ty}): layout errored ({err:?}), yet \
                              tail `{tail}` is not a type parameter or a projection",
                    ),
                }
            }
            ty::Array(inner, len)
                if len.ty() == tcx.types.usize && tcx.features().transmute_generic_consts =>
            {
                match SizeSkeleton::compute(inner, tcx, param_env)? {
                    // This may succeed because the multiplication of two types may overflow
                    // but a single size of a nested array will not.
                    SizeSkeleton::Known(s) => {
                        if let Some(c) = len.try_eval_target_usize(tcx, param_env) {
                            let size = s
                                .bytes()
                                .checked_mul(c)
                                .ok_or_else(|| &*tcx.arena.alloc(LayoutError::SizeOverflow(ty)))?;
                            return Ok(SizeSkeleton::Known(Size::from_bytes(size)));
                        }
                        let len = tcx.expand_abstract_consts(len);
                        let prev = ty::Const::from_target_usize(tcx, s.bytes());
                        let Some(gen_size) = mul_sorted_consts(tcx, param_env, len, prev) else {
                            return Err(tcx.arena.alloc(LayoutError::SizeOverflow(ty)));
                        };
                        Ok(SizeSkeleton::Generic(gen_size))
                    }
                    SizeSkeleton::Pointer { .. } => Err(err),
                    SizeSkeleton::Generic(g) => {
                        let len = tcx.expand_abstract_consts(len);
                        let Some(gen_size) = mul_sorted_consts(tcx, param_env, len, g) else {
                            return Err(tcx.arena.alloc(LayoutError::SizeOverflow(ty)));
                        };
                        Ok(SizeSkeleton::Generic(gen_size))
                    }
                }
            }

            ty::Adt(def, substs) => {
                // Only newtypes and enums w/ nullable pointer optimization.
                if def.is_union() || def.variants().is_empty() || def.variants().len() > 2 {
                    return Err(err);
                }

                // Get a zero-sized variant or a pointer newtype.
                let zero_or_ptr_variant = |i| {
                    let i = VariantIdx::from_usize(i);
                    let fields =
                        def.variant(i).fields.iter().map(|field| {
                            SizeSkeleton::compute(field.ty(tcx, substs), tcx, param_env)
                        });
                    let mut ptr = None;
                    for field in fields {
                        let field = field?;
                        match field {
                            SizeSkeleton::Known(size) => {
                                if size.bytes() > 0 {
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
                let normalized = tcx.normalize_erasing_regions(param_env, ty);
                if ty == normalized {
                    Err(err)
                } else {
                    SizeSkeleton::compute(normalized, tcx, param_env)
                }
            }

            _ => Err(err),
        }
    }

    pub fn same_size(self, other: SizeSkeleton<'tcx>) -> bool {
        match (self, other) {
            (SizeSkeleton::Known(a), SizeSkeleton::Known(b)) => a == b,
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

/// When creating the layout for types with abstract consts in their size (i.e. [usize; 4 * N]),
/// to ensure that they have a canonical order and can be compared directly we combine all
/// constants, and sort the other terms. This allows comparison of expressions of sizes,
/// allowing for things like transmuting between types that depend on generic consts.
/// This returns `None` if multiplication of constants overflows.
fn mul_sorted_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    a: ty::Const<'tcx>,
    b: ty::Const<'tcx>,
) -> Option<ty::Const<'tcx>> {
    use crate::mir::BinOp::Mul;

    let mut work = vec![a, b];
    let mut done = vec![];
    while let Some(n) = work.pop() {
        if let ConstKind::Expr(ty::Expr::Binop(Mul, l, r)) = n.kind() {
            work.push(l);
            work.push(r)
        } else {
            done.push(n);
        }
    }
    let mut k = 1;
    let mut overflow = false;
    done.retain(|c| {
        let Some(c) = c.try_eval_target_usize(tcx, param_env) else {
            return true;
        };
        let Some(next) = c.checked_mul(k) else {
            overflow = true;
            return false;
        };
        k = next;
        false
    });
    if overflow {
        return None;
    }
    if k != 1 {
        done.push(ty::Const::from_target_usize(tcx, k));
    } else if k == 0 {
        return Some(ty::Const::from_target_usize(tcx, 0));
    }
    done.sort_unstable();

    // create a single tree from the buffer
    done.into_iter().reduce(|acc, n| ty::Const::new_expr(tcx, ty::Expr::Binop(Mul, n, acc), n.ty()))
}

pub trait HasTyCtxt<'tcx>: HasDataLayout {
    fn tcx(&self) -> TyCtxt<'tcx>;
}

pub trait HasParamEnv<'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx>;
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

impl<'tcx, C> HasParamEnv<'tcx> for LayoutCx<'tcx, C> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'tcx, T: HasDataLayout> HasDataLayout for LayoutCx<'tcx, T> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.tcx.data_layout()
    }
}

impl<'tcx, T: HasTargetSpec> HasTargetSpec for LayoutCx<'tcx, T> {
    fn target_spec(&self) -> &Target {
        self.tcx.target_spec()
    }
}

impl<'tcx, T: HasTyCtxt<'tcx>> HasTyCtxt<'tcx> for LayoutCx<'tcx, T> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx.tcx()
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

pub type TyAndLayout<'tcx> = rustc_target::abi::TyAndLayout<'tcx, Ty<'tcx>>;

/// Trait for contexts that want to be able to compute layouts of types.
/// This automatically gives access to `LayoutOf`, through a blanket `impl`.
pub trait LayoutOfHelpers<'tcx>: HasDataLayout + HasTyCtxt<'tcx> + HasParamEnv<'tcx> {
    /// The `TyAndLayout`-wrapping type (or `TyAndLayout` itself), which will be
    /// returned from `layout_of` (see also `handle_layout_err`).
    type LayoutOfResult: MaybeResult<TyAndLayout<'tcx>>;

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
    /// executes in "reveal all" mode, and will normalize the input type.
    #[inline]
    fn layout_of(&self, ty: Ty<'tcx>) -> Self::LayoutOfResult {
        self.spanned_layout_of(ty, DUMMY_SP)
    }

    /// Computes the layout of a type, at `span`. Note that this implicitly
    /// executes in "reveal all" mode, and will normalize the input type.
    // FIXME(eddyb) avoid passing information like this, and instead add more
    // `TyCtxt::at`-like APIs to be able to do e.g. `cx.at(span).layout_of(ty)`.
    #[inline]
    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::LayoutOfResult {
        let span = if !span.is_dummy() { span } else { self.layout_tcx_at_span() };
        let tcx = self.tcx().at(span);

        MaybeResult::from(
            tcx.layout_of(self.param_env().and(ty))
                .map_err(|err| self.handle_layout_err(*err, span, ty)),
        )
    }
}

impl<'tcx, C: LayoutOfHelpers<'tcx>> LayoutOf<'tcx> for C {}

impl<'tcx> LayoutOfHelpers<'tcx> for LayoutCx<'tcx, TyCtxt<'tcx>> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, &'tcx LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        _: Span,
        _: Ty<'tcx>,
    ) -> &'tcx LayoutError<'tcx> {
        self.tcx.arena.alloc(err)
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for LayoutCx<'tcx, TyCtxtAt<'tcx>> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, &'tcx LayoutError<'tcx>>;

    #[inline]
    fn layout_tcx_at_span(&self) -> Span {
        self.tcx.span
    }

    #[inline]
    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        _: Span,
        _: Ty<'tcx>,
    ) -> &'tcx LayoutError<'tcx> {
        self.tcx.arena.alloc(err)
    }
}

impl<'tcx, C> TyAbiInterface<'tcx, C> for Ty<'tcx>
where
    C: HasTyCtxt<'tcx> + HasParamEnv<'tcx>,
{
    fn ty_and_layout_for_variant(
        this: TyAndLayout<'tcx>,
        cx: &C,
        variant_index: VariantIdx,
    ) -> TyAndLayout<'tcx> {
        let layout = match this.variants {
            Variants::Single { index }
                // If all variants but one are uninhabited, the variant layout is the enum layout.
                if index == variant_index &&
                // Don't confuse variants of uninhabited enums with the enum itself.
                // For more details see https://github.com/rust-lang/rust/issues/69763.
                this.fields != FieldsShape::Primitive =>
            {
                this.layout
            }

            Variants::Single { index } => {
                let tcx = cx.tcx();
                let param_env = cx.param_env();

                // Deny calling for_variant more than once for non-Single enums.
                if let Ok(original_layout) = tcx.layout_of(param_env.and(this.ty)) {
                    assert_eq!(original_layout.variants, Variants::Single { index });
                }

                let fields = match this.ty.kind() {
                    ty::Adt(def, _) if def.variants().is_empty() =>
                        bug!("for_variant called on zero-variant enum"),
                    ty::Adt(def, _) => def.variant(variant_index).fields.len(),
                    _ => bug!(),
                };
                tcx.mk_layout(LayoutS {
                    variants: Variants::Single { index: variant_index },
                    fields: match NonZeroUsize::new(fields) {
                        Some(fields) => FieldsShape::Union(fields),
                        None => FieldsShape::Arbitrary { offsets: IndexVec::new(), memory_index: IndexVec::new() },
                    },
                    abi: Abi::Uninhabited,
                    largest_niche: None,
                    align: tcx.data_layout.i8_align,
                    size: Size::ZERO,
                })
            }

            Variants::Multiple { ref variants, .. } => cx.tcx().mk_layout(variants[variant_index].clone()),
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
            cx: &(impl HasTyCtxt<'tcx> + HasParamEnv<'tcx>),
            i: usize,
        ) -> TyMaybeWithLayout<'tcx> {
            let tcx = cx.tcx();
            let tag_layout = |tag: Scalar| -> TyAndLayout<'tcx> {
                TyAndLayout {
                    layout: tcx.mk_layout(LayoutS::scalar(cx, tag)),
                    ty: tag.primitive().to_ty(tcx),
                }
            };

            match *this.ty.kind() {
                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::FnPtr(_)
                | ty::Never
                | ty::FnDef(..)
                | ty::GeneratorWitness(..)
                | ty::GeneratorWitnessMIR(..)
                | ty::Foreign(..)
                | ty::Dynamic(_, _, ty::Dyn) => {
                    bug!("TyAndLayout::field({:?}): not applicable", this)
                }

                // Potentially-fat pointers.
                ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                    assert!(i < this.fields.count());

                    // Reuse the fat `*T` type as its own thin pointer data field.
                    // This provides information about, e.g., DST struct pointees
                    // (which may have no non-DST form), and will work as long
                    // as the `Abi` or `FieldsShape` is checked by users.
                    if i == 0 {
                        let nil = Ty::new_unit(tcx);
                        let unit_ptr_ty = if this.ty.is_unsafe_ptr() {
                            Ty::new_mut_ptr(tcx, nil)
                        } else {
                            Ty::new_mut_ref(tcx, tcx.lifetimes.re_static, nil)
                        };

                        // NOTE(eddyb) using an empty `ParamEnv`, and `unwrap`-ing
                        // the `Result` should always work because the type is
                        // always either `*mut ()` or `&'static mut ()`.
                        return TyMaybeWithLayout::TyAndLayout(TyAndLayout {
                            ty: this.ty,
                            ..tcx.layout_of(ty::ParamEnv::reveal_all().and(unit_ptr_ty)).unwrap()
                        });
                    }

                    let mk_dyn_vtable = || {
                        Ty::new_imm_ref(
                            tcx,
                            tcx.lifetimes.re_static,
                            Ty::new_array(tcx, tcx.types.usize, 3),
                        )
                        /* FIXME: use actual fn pointers
                        Warning: naively computing the number of entries in the
                        vtable by counting the methods on the trait + methods on
                        all parent traits does not work, because some methods can
                        be not object safe and thus excluded from the vtable.
                        Increase this counter if you tried to implement this but
                        failed to do it without duplicating a lot of code from
                        other places in the compiler: 2
                        Ty::new_tup(tcx,&[
                            Ty::new_array(tcx,tcx.types.usize, 3),
                            Ty::new_array(tcx,Option<fn()>),
                        ])
                        */
                    };

                    let metadata = if let Some(metadata_def_id) = tcx.lang_items().metadata_type()
                        // Projection eagerly bails out when the pointee references errors,
                        // fall back to structurally deducing metadata.
                        && !pointee.references_error()
                    {
                        let metadata = tcx.normalize_erasing_regions(
                            cx.param_env(),
                            Ty::new_projection(tcx,metadata_def_id, [pointee]),
                        );

                        // Map `Metadata = DynMetadata<dyn Trait>` back to a vtable, since it
                        // offers better information than `std::ptr::metadata::VTable`,
                        // and we rely on this layout information to trigger a panic in
                        // `std::mem::uninitialized::<&dyn Trait>()`, for example.
                        if let ty::Adt(def, substs) = metadata.kind()
                            && Some(def.did()) == tcx.lang_items().dyn_metadata()
                            && substs.type_at(0).is_trait()
                        {
                            mk_dyn_vtable()
                        } else {
                            metadata
                        }
                    } else {
                        match tcx.struct_tail_erasing_lifetimes(pointee, cx.param_env()).kind() {
                            ty::Slice(_) | ty::Str => tcx.types.usize,
                            ty::Dynamic(_, _, ty::Dyn) => mk_dyn_vtable(),
                            _ => bug!("TyAndLayout::field({:?}): not applicable", this),
                        }
                    };

                    TyMaybeWithLayout::Ty(metadata)
                }

                // Arrays and slices.
                ty::Array(element, _) | ty::Slice(element) => TyMaybeWithLayout::Ty(element),
                ty::Str => TyMaybeWithLayout::Ty(tcx.types.u8),

                // Tuples, generators and closures.
                ty::Closure(_, ref substs) => field_ty_or_layout(
                    TyAndLayout { ty: substs.as_closure().tupled_upvars_ty(), ..this },
                    cx,
                    i,
                ),

                ty::Generator(def_id, ref substs, _) => match this.variants {
                    Variants::Single { index } => TyMaybeWithLayout::Ty(
                        substs
                            .as_generator()
                            .state_tys(def_id, tcx)
                            .nth(index.as_usize())
                            .unwrap()
                            .nth(i)
                            .unwrap(),
                    ),
                    Variants::Multiple { tag, tag_field, .. } => {
                        if i == tag_field {
                            return TyMaybeWithLayout::TyAndLayout(tag_layout(tag));
                        }
                        TyMaybeWithLayout::Ty(substs.as_generator().prefix_tys().nth(i).unwrap())
                    }
                },

                ty::Tuple(tys) => TyMaybeWithLayout::Ty(tys[i]),

                // ADTs.
                ty::Adt(def, substs) => {
                    match this.variants {
                        Variants::Single { index } => {
                            let field = &def.variant(index).fields[FieldIdx::from_usize(i)];
                            TyMaybeWithLayout::Ty(field.ty(tcx, substs))
                        }

                        // Discriminant field for enums (where applicable).
                        Variants::Multiple { tag, .. } => {
                            assert_eq!(i, 0);
                            return TyMaybeWithLayout::TyAndLayout(tag_layout(tag));
                        }
                    }
                }

                ty::Dynamic(_, _, ty::DynStar) => {
                    if i == 0 {
                        TyMaybeWithLayout::Ty(Ty::new_mut_ptr(tcx, tcx.types.unit))
                    } else if i == 1 {
                        // FIXME(dyn-star) same FIXME as above applies here too
                        TyMaybeWithLayout::Ty(Ty::new_imm_ref(
                            tcx,
                            tcx.lifetimes.re_static,
                            Ty::new_array(tcx, tcx.types.usize, 3),
                        ))
                    } else {
                        bug!("no field {i} on dyn*")
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
                cx.tcx().layout_of(cx.param_env().and(field_ty)).unwrap_or_else(|e| {
                    bug!(
                        "failed to get layout for `{field_ty}`: {e:?},\n\
                         despite it being a field (#{i}) of an existing layout: {this:#?}",
                    )
                })
            }
            TyMaybeWithLayout::TyAndLayout(field_layout) => field_layout,
        }
    }

    fn ty_and_layout_pointee_info_at(
        this: TyAndLayout<'tcx>,
        cx: &C,
        offset: Size,
    ) -> Option<PointeeInfo> {
        let tcx = cx.tcx();
        let param_env = cx.param_env();

        let pointee_info = match *this.ty.kind() {
            ty::RawPtr(mt) if offset.bytes() == 0 => {
                tcx.layout_of(param_env.and(mt.ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: None,
                })
            }
            ty::FnPtr(fn_sig) if offset.bytes() == 0 => {
                tcx.layout_of(param_env.and(Ty::new_fn_ptr(tcx, fn_sig))).ok().map(|layout| {
                    PointeeInfo { size: layout.size, align: layout.align.abi, safe: None }
                })
            }
            ty::Ref(_, ty, mt) if offset.bytes() == 0 => {
                // Use conservative pointer kind if not optimizing. This saves us the
                // Freeze/Unpin queries, and can save time in the codegen backend (noalias
                // attributes in LLVM have compile-time cost even in unoptimized builds).
                let optimize = tcx.sess.opts.optimize != OptLevel::No;
                let kind = match mt {
                    hir::Mutability::Not => PointerKind::SharedRef {
                        frozen: optimize && ty.is_freeze(tcx, cx.param_env()),
                    },
                    hir::Mutability::Mut => PointerKind::MutableRef {
                        unpin: optimize && ty.is_unpin(tcx, cx.param_env()),
                    },
                };

                tcx.layout_of(param_env.and(ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: Some(kind),
                })
            }

            _ => {
                let mut data_variant = match this.variants {
                    // Within the discriminant field, only the niche itself is
                    // always initialized, so we only check for a pointer at its
                    // offset.
                    //
                    // If the niche is a pointer, it's either valid (according
                    // to its type), or null (which the niche field's scalar
                    // validity range encodes). This allows using
                    // `dereferenceable_or_null` for e.g., `Option<&T>`, and
                    // this will continue to work as long as we don't start
                    // using more niches than just null (e.g., the first page of
                    // the address space, or unaligned pointers).
                    Variants::Multiple {
                        tag_encoding: TagEncoding::Niche { untagged_variant, .. },
                        tag_field,
                        ..
                    } if this.fields.offset(tag_field) == offset => {
                        Some(this.for_variant(cx, untagged_variant))
                    }
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
                    let ptr_end = offset + Pointer(AddressSpace::DATA).size(cx);
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

                // FIXME(eddyb) This should be for `ptr::Unique<T>`, not `Box<T>`.
                if let Some(ref mut pointee) = result {
                    if let ty::Adt(def, _) = this.ty.kind() {
                        if def.is_box() && offset.bytes() == 0 {
                            let optimize = tcx.sess.opts.optimize != OptLevel::No;
                            pointee.safe = Some(PointerKind::Box {
                                unpin: optimize && this.ty.boxed_ty().is_unpin(tcx, cx.param_env()),
                            });
                        }
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
        this.ty.kind() == &ty::Never
    }

    fn is_tuple(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Tuple(..))
    }

    fn is_unit(this: TyAndLayout<'tcx>) -> bool {
        matches!(this.ty.kind(), ty::Tuple(list) if list.len() == 0)
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
pub fn fn_can_unwind(tcx: TyCtxt<'_>, fn_def_id: Option<DefId>, abi: SpecAbi) -> bool {
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
        if tcx.sess.opts.unstable_opts.panic_in_drop == PanicStrategy::Abort {
            if Some(did) == tcx.lang_items().drop_in_place_fn() {
                return false;
            }
        }
    }

    // Otherwise if this isn't special then unwinding is generally determined by
    // the ABI of the itself. ABIs like `C` have variants which also
    // specifically allow unwinding (`C-unwind`), but not all platform-specific
    // ABIs have such an option. Otherwise the only other thing here is Rust
    // itself, and those ABIs are determined by the panic strategy configured
    // for this compilation.
    //
    // Unfortunately at this time there's also another caveat. Rust [RFC
    // 2945][rfc] has been accepted and is in the process of being implemented
    // and stabilized. In this interim state we need to deal with historical
    // rustc behavior as well as plan for future rustc behavior.
    //
    // Historically functions declared with `extern "C"` were marked at the
    // codegen layer as `nounwind`. This happened regardless of `panic=unwind`
    // or not. This is UB for functions in `panic=unwind` mode that then
    // actually panic and unwind. Note that this behavior is true for both
    // externally declared functions as well as Rust-defined function.
    //
    // To fix this UB rustc would like to change in the future to catch unwinds
    // from function calls that may unwind within a Rust-defined `extern "C"`
    // function and forcibly abort the process, thereby respecting the
    // `nounwind` attribute emitted for `extern "C"`. This behavior change isn't
    // ready to roll out, so determining whether or not the `C` family of ABIs
    // unwinds is conditional not only on their definition but also whether the
    // `#![feature(c_unwind)]` feature gate is active.
    //
    // Note that this means that unlike historical compilers rustc now, by
    // default, unconditionally thinks that the `C` ABI may unwind. This will
    // prevent some optimization opportunities, however, so we try to scope this
    // change and only assume that `C` unwinds with `panic=unwind` (as opposed
    // to `panic=abort`).
    //
    // Eventually the check against `c_unwind` here will ideally get removed and
    // this'll be a little cleaner as it'll be a straightforward check of the
    // ABI.
    //
    // [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/2945-c-unwind-abi.md
    use SpecAbi::*;
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
        | SysV64 { unwind } => {
            unwind
                || (!tcx.features().c_unwind && tcx.sess.panic_strategy() == PanicStrategy::Unwind)
        }
        PtxKernel
        | Msp430Interrupt
        | X86Interrupt
        | AmdGpuKernel
        | EfiApi
        | AvrInterrupt
        | AvrNonBlockingInterrupt
        | CCmseNonSecureCall
        | Wasm
        | PlatformIntrinsic
        | Unadjusted => false,
        Rust | RustCall | RustCold | RustIntrinsic => {
            tcx.sess.panic_strategy() == PanicStrategy::Unwind
        }
    }
}

/// Error produced by attempting to compute or adjust a `FnAbi`.
#[derive(Copy, Clone, Debug, HashStable)]
pub enum FnAbiError<'tcx> {
    /// Error produced by a `layout_of` call, while computing `FnAbi` initially.
    Layout(LayoutError<'tcx>),

    /// Error produced by attempting to adjust a `FnAbi`, for a "foreign" ABI.
    AdjustForForeignAbi(call::AdjustForForeignAbiError),
}

impl<'a, 'b> IntoDiagnostic<'a, !> for FnAbiError<'b> {
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'a, !> {
        match self {
            Self::Layout(e) => e.into_diagnostic().into_diagnostic(handler),
            Self::AdjustForForeignAbi(call::AdjustForForeignAbiError::Unsupported {
                arch,
                abi,
            }) => UnsupportedFnAbi { arch, abi: abi.name() }.into_diagnostic(handler),
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
    type FnAbiOfResult: MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>>;

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
    /// instead, where the instance is an `InstanceDef::Virtual`.
    #[inline]
    fn fn_abi_of_fn_ptr(
        &self,
        sig: ty::PolyFnSig<'tcx>,
        extra_args: &'tcx ty::List<Ty<'tcx>>,
    ) -> Self::FnAbiOfResult {
        // FIXME(eddyb) get a better `span` here.
        let span = self.layout_tcx_at_span();
        let tcx = self.tcx().at(span);

        MaybeResult::from(tcx.fn_abi_of_fn_ptr(self.param_env().and((sig, extra_args))).map_err(
            |err| self.handle_fn_abi_err(*err, span, FnAbiRequest::OfFnPtr { sig, extra_args }),
        ))
    }

    /// Compute a `FnAbi` suitable for declaring/defining an `fn` instance, and for
    /// direct calls to an `fn`.
    ///
    /// NB: that includes virtual calls, which are represented by "direct calls"
    /// to an `InstanceDef::Virtual` instance (of `<dyn Trait as Trait>::fn`).
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
            tcx.fn_abi_of_instance(self.param_env().and((instance, extra_args))).map_err(|err| {
                // HACK(eddyb) at least for definitions of/calls to `Instance`s,
                // we can get some kind of span even if one wasn't provided.
                // However, we don't do this early in order to avoid calling
                // `def_span` unconditionally (which may have a perf penalty).
                let span = if !span.is_dummy() { span } else { tcx.def_span(instance.def_id()) };
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
