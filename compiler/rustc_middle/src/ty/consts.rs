use std::borrow::Cow;

use rustc_data_structures::intern::Interned;
use rustc_error_messages::MultiSpan;
use rustc_macros::HashStable;
use rustc_type_ir::{self as ir, TypeFlags, WithCachedTypeInfo};

use crate::mir::interpret::Scalar;
use crate::ty::{self, Ty, TyCtxt};

mod int;
mod kind;
mod valtree;

pub use int::*;
pub use kind::*;
use rustc_span::{DUMMY_SP, ErrorGuaranteed};
pub use valtree::*;

pub type ConstKind<'tcx> = ir::ConstKind<TyCtxt<'tcx>>;
pub type UnevaluatedConst<'tcx> = ir::UnevaluatedConst<TyCtxt<'tcx>>;

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(ConstKind<'_>, 32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Const<'tcx>(pub(super) Interned<'tcx, WithCachedTypeInfo<ConstKind<'tcx>>>);

impl<'tcx> rustc_type_ir::inherent::IntoKind for Const<'tcx> {
    type Kind = ConstKind<'tcx>;

    fn kind(self) -> ConstKind<'tcx> {
        self.kind()
    }
}

impl<'tcx> rustc_type_ir::visit::Flags for Const<'tcx> {
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl<'tcx> Const<'tcx> {
    #[inline]
    pub fn kind(self) -> ConstKind<'tcx> {
        let a: &ConstKind<'tcx> = self.0.0;
        *a
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline]
    pub fn outer_exclusive_binder(self) -> ty::DebruijnIndex {
        self.0.outer_exclusive_binder
    }

    #[inline]
    pub fn new(tcx: TyCtxt<'tcx>, kind: ty::ConstKind<'tcx>) -> Const<'tcx> {
        tcx.mk_ct_from_kind(kind)
    }

    #[inline]
    pub fn new_param(tcx: TyCtxt<'tcx>, param: ty::ParamConst) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Param(param))
    }

    #[inline]
    pub fn new_var(tcx: TyCtxt<'tcx>, infer: ty::ConstVid) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Var(infer)))
    }

    #[inline]
    pub fn new_fresh(tcx: TyCtxt<'tcx>, fresh: u32) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Fresh(fresh)))
    }

    #[inline]
    pub fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(infer))
    }

    #[inline]
    pub fn new_bound(
        tcx: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        var: ty::BoundVar,
    ) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Bound(debruijn, var))
    }

    #[inline]
    pub fn new_placeholder(tcx: TyCtxt<'tcx>, placeholder: ty::PlaceholderConst) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Placeholder(placeholder))
    }

    #[inline]
    pub fn new_unevaluated(tcx: TyCtxt<'tcx>, uv: ty::UnevaluatedConst<'tcx>) -> Const<'tcx> {
        tcx.debug_assert_args_compatible(uv.def, uv.args);
        Const::new(tcx, ty::ConstKind::Unevaluated(uv))
    }

    #[inline]
    pub fn new_value(tcx: TyCtxt<'tcx>, val: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Value(ty, val))
    }

    #[inline]
    pub fn new_expr(tcx: TyCtxt<'tcx>, expr: ty::Expr<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Expr(expr))
    }

    #[inline]
    pub fn new_error(tcx: TyCtxt<'tcx>, e: ty::ErrorGuaranteed) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Error(e))
    }

    /// Like [Ty::new_error] but for constants.
    #[track_caller]
    pub fn new_misc_error(tcx: TyCtxt<'tcx>) -> Const<'tcx> {
        Const::new_error_with_message(
            tcx,
            DUMMY_SP,
            "ty::ConstKind::Error constructed but no error reported",
        )
    }

    /// Like [Ty::new_error_with_message] but for constants.
    #[track_caller]
    pub fn new_error_with_message<S: Into<MultiSpan>>(
        tcx: TyCtxt<'tcx>,
        span: S,
        msg: impl Into<Cow<'static, str>>,
    ) -> Const<'tcx> {
        let reported = tcx.dcx().span_delayed_bug(span, msg);
        Const::new_error(tcx, reported)
    }
}

impl<'tcx> rustc_type_ir::inherent::Const<TyCtxt<'tcx>> for Const<'tcx> {
    fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst) -> Self {
        Const::new_infer(tcx, infer)
    }

    fn new_var(tcx: TyCtxt<'tcx>, vid: ty::ConstVid) -> Self {
        Const::new_var(tcx, vid)
    }

    fn new_bound(interner: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Const::new_bound(interner, debruijn, var)
    }

    fn new_anon_bound(tcx: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Const::new_bound(tcx, debruijn, var)
    }

    fn new_unevaluated(interner: TyCtxt<'tcx>, uv: ty::UnevaluatedConst<'tcx>) -> Self {
        Const::new_unevaluated(interner, uv)
    }

    fn new_expr(interner: TyCtxt<'tcx>, expr: ty::Expr<'tcx>) -> Self {
        Const::new_expr(interner, expr)
    }

    fn new_error(interner: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> Self {
        Const::new_error(interner, guar)
    }
}

impl<'tcx> Const<'tcx> {
    /// Creates a constant with the given integer value and interns it.
    #[inline]
    pub fn from_bits(
        tcx: TyCtxt<'tcx>,
        bits: u128,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Self {
        let size = tcx
            .layout_of(typing_env.as_query_input(ty))
            .unwrap_or_else(|e| panic!("could not compute layout for {ty:?}: {e:?}"))
            .size;
        ty::Const::new_value(
            tcx,
            ty::ValTree::from_scalar_int(ScalarInt::try_from_uint(bits, size).unwrap()),
            ty,
        )
    }

    #[inline]
    /// Creates an interned zst constant.
    pub fn zero_sized(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        ty::Const::new_value(tcx, ty::ValTree::zst(), ty)
    }

    #[inline]
    /// Creates an interned bool constant.
    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        Self::from_bits(tcx, v as u128, ty::TypingEnv::fully_monomorphized(), tcx.types.bool)
    }

    #[inline]
    /// Creates an interned usize constant.
    pub fn from_target_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        Self::from_bits(tcx, n as u128, ty::TypingEnv::fully_monomorphized(), tcx.types.usize)
    }

    /// Panics if self.kind != ty::ConstKind::Value
    pub fn to_valtree(self) -> (ty::ValTree<'tcx>, Ty<'tcx>) {
        match self.kind() {
            ty::ConstKind::Value(ty, valtree) => (valtree, ty),
            _ => bug!("expected ConstKind::Value, got {:?}", self.kind()),
        }
    }

    /// Attempts to convert to a `ValTree`
    pub fn try_to_valtree(self) -> Option<(ty::ValTree<'tcx>, Ty<'tcx>)> {
        match self.kind() {
            ty::ConstKind::Value(ty, valtree) => Some((valtree, ty)),
            _ => None,
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<(Scalar, Ty<'tcx>)> {
        let (valtree, ty) = self.try_to_valtree()?;
        Some((valtree.try_to_scalar()?, ty))
    }

    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_valtree()?.0.try_to_scalar_int()?.try_to_bool().ok()
    }

    #[inline]
    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_valtree()?.0.try_to_target_usize(tcx)
    }

    /// Attempts to evaluate the given constant to bits. Can fail to evaluate in the presence of
    /// generics (or erroneous code) or if the value can't be represented as bits (e.g. because it
    /// contains const generic parameters or pointers).
    #[inline]
    pub fn try_to_bits(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> Option<u128> {
        let (scalar, ty) = self.try_to_scalar()?;
        let scalar = scalar.try_to_scalar_int().ok()?;
        let input = typing_env.with_post_analysis_normalized(tcx).as_query_input(ty);
        let size = tcx.layout_of(input).ok()?.size;
        Some(scalar.to_bits(size))
    }

    pub fn is_ct_infer(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(_))
    }
}
