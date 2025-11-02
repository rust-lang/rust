use std::borrow::Cow;

use rustc_data_structures::intern::Interned;
use rustc_error_messages::MultiSpan;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_type_ir::walk::TypeWalker;
use rustc_type_ir::{self as ir, TypeFlags, WithCachedTypeInfo};

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
rustc_data_structures::static_assert_size!(ConstKind<'_>, 24);

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Const<'tcx>(pub(super) Interned<'tcx, WithCachedTypeInfo<ConstKind<'tcx>>>);

impl<'tcx> rustc_type_ir::inherent::IntoKind for Const<'tcx> {
    type Kind = ConstKind<'tcx>;

    fn kind(self) -> ConstKind<'tcx> {
        self.kind()
    }
}

impl<'tcx> rustc_type_ir::Flags for Const<'tcx> {
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
        bound_const: ty::BoundConst,
    ) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_const))
    }

    #[inline]
    pub fn new_canonical_bound(tcx: TyCtxt<'tcx>, var: ty::BoundVar) -> Const<'tcx> {
        Const::new(
            tcx,
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Canonical, ty::BoundConst { var }),
        )
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
    pub fn new_value(tcx: TyCtxt<'tcx>, valtree: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Value(ty::Value { ty, valtree }))
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

    pub fn is_trivially_wf(self) -> bool {
        match self.kind() {
            ty::ConstKind::Param(_) | ty::ConstKind::Placeholder(_) | ty::ConstKind::Bound(..) => {
                true
            }
            ty::ConstKind::Infer(_)
            | ty::ConstKind::Unevaluated(..)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Error(_)
            | ty::ConstKind::Expr(_) => false,
        }
    }
}

impl<'tcx> rustc_type_ir::inherent::Const<TyCtxt<'tcx>> for Const<'tcx> {
    fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst) -> Self {
        Const::new_infer(tcx, infer)
    }

    fn new_var(tcx: TyCtxt<'tcx>, vid: ty::ConstVid) -> Self {
        Const::new_var(tcx, vid)
    }

    fn new_bound(
        interner: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        bound_const: ty::BoundConst,
    ) -> Self {
        Const::new_bound(interner, debruijn, bound_const)
    }

    fn new_anon_bound(tcx: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Const::new_bound(tcx, debruijn, ty::BoundConst { var })
    }

    fn new_canonical_bound(tcx: TyCtxt<'tcx>, var: rustc_type_ir::BoundVar) -> Self {
        Const::new_canonical_bound(tcx, var)
    }

    fn new_placeholder(tcx: TyCtxt<'tcx>, placeholder: ty::PlaceholderConst) -> Self {
        Const::new_placeholder(tcx, placeholder)
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
            ty::ValTree::from_scalar_int(tcx, ScalarInt::try_from_uint(bits, size).unwrap()),
            ty,
        )
    }

    #[inline]
    /// Creates an interned zst constant.
    pub fn zero_sized(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        ty::Const::new_value(tcx, ty::ValTree::zst(tcx), ty)
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

    /// Panics if `self.kind != ty::ConstKind::Value`.
    pub fn to_value(self) -> ty::Value<'tcx> {
        match self.kind() {
            ty::ConstKind::Value(cv) => cv,
            _ => bug!("expected ConstKind::Value, got {:?}", self.kind()),
        }
    }

    /// Attempts to convert to a value.
    ///
    /// Note that this does not evaluate the constant.
    pub fn try_to_value(self) -> Option<ty::Value<'tcx>> {
        match self.kind() {
            ty::ConstKind::Value(cv) => Some(cv),
            _ => None,
        }
    }

    /// Convenience method to extract the value of a usize constant,
    /// useful to get the length of an array type.
    ///
    /// Note that this does not evaluate the constant.
    #[inline]
    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_value()?.try_to_target_usize(tcx)
    }

    pub fn is_ct_infer(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(_))
    }

    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(self) -> TypeWalker<TyCtxt<'tcx>> {
        TypeWalker::new(self.into())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum AnonConstKind {
    /// `feature(generic_const_exprs)` anon consts are allowed to use arbitrary generic parameters in scope
    GCE,
    /// stable `min_const_generics` anon consts are not allowed to use any generic parameters
    MCG,
    /// anon consts used as the length of a repeat expr are syntactically allowed to use generic parameters
    /// but must not depend on the actual instantiation. See #76200 for more information
    RepeatExprCount,
    /// anon consts outside of the type system, e.g. enum discriminants
    NonTypeSystem,
}
