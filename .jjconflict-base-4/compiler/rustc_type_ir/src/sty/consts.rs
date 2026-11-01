use std::fmt;

use derive_where::derive_where;
use rustc_ast_ir::visit::VisitorResult;
#[cfg(feature = "nightly")]
use rustc_macros::StableHash_NoContext;
use rustc_type_ir_macros::{GenericTypeVisitable, Lift_Generic};

use crate::inherent::*;
use crate::intern::Interned;
use crate::relate::{Relate, RelateResult, TypeRelation};
use crate::{
    AliasConst, BoundConst, BoundVar, BoundVarIndexKind, ConstKind, ConstVid, DebruijnIndex,
    FallibleTypeFolder, Flags, InferConst, Interner, IsRigid, PlaceholderConst, TypeFlags,
    TypeFoldable, TypeFolder, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
#[derive(GenericTypeVisitable, Lift_Generic)]
pub struct Const<I: Interner>(pub I::InternedConstKind);

impl<I: Interner> fmt::Debug for Const<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If this is a value, we spend some effort to make it look nice.
        if let ConstKind::Value(cv) = self.kind() {
            write!(f, "{}", cv)
        } else {
            // Fall back to something verbose.
            write!(f, "{:?}", self.kind())
        }
    }
}

impl<I: Interner> Const<I> {
    #[inline]
    pub fn kind(self) -> ConstKind<I> {
        *self.0.get()
    }

    #[inline]
    pub fn new(interner: I, kind: ConstKind<I>) -> Self {
        interner.mk_ct_from_kind(kind)
    }

    #[inline]
    pub fn new_var(interner: I, infer: ConstVid) -> Self {
        Self::new(interner, ConstKind::Infer(InferConst::Var(infer)))
    }

    #[inline]
    pub fn new_infer(interner: I, infer: InferConst) -> Self {
        Self::new(interner, ConstKind::Infer(infer))
    }

    #[inline]
    pub fn new_bound(interner: I, debruijn: DebruijnIndex, bound_const: BoundConst<I>) -> Self {
        Self::new(interner, ConstKind::Bound(BoundVarIndexKind::Bound(debruijn), bound_const))
    }

    #[inline]
    pub fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Self::new_bound(interner, debruijn, BoundConst::new(var))
    }

    #[inline]
    pub fn new_canonical_bound(interner: I, var: BoundVar) -> Self {
        Self::new(interner, ConstKind::Bound(BoundVarIndexKind::Canonical, BoundConst::new(var)))
    }

    #[inline]
    pub fn new_placeholder(interner: I, placeholder: PlaceholderConst<I>) -> Self {
        Self::new(interner, ConstKind::Placeholder(placeholder))
    }

    #[inline]
    pub fn new_alias(interner: I, is_rigid: IsRigid, alias_const: AliasConst<I>) -> Self {
        Self::new(interner, ConstKind::Alias(is_rigid, alias_const))
    }

    #[inline]
    pub fn new_expr(interner: I, expr: I::ExprConst) -> Self {
        Const::new(interner, ConstKind::Expr(expr))
    }

    #[inline]
    pub fn new_error(interner: I, e: I::ErrorGuaranteed) -> Self {
        Const::new(interner, ConstKind::Error(e))
    }

    #[inline]
    pub fn is_ct_var(self) -> bool {
        matches!(self.kind(), ConstKind::Infer(InferConst::Var(_)))
    }

    #[inline]
    pub fn is_ct_error(self) -> bool {
        matches!(self.kind(), ConstKind::Error(_))
    }

    #[inline]
    pub fn new_param(interner: I, param: I::ParamConst) -> Self {
        Self::new(interner, ConstKind::Param(param))
    }

    #[inline]
    pub fn new_fresh(interner: I, fresh: u32) -> Self {
        Self::new(interner, ConstKind::Infer(InferConst::Fresh(fresh)))
    }

    #[track_caller]
    pub fn new_misc_error(interner: I) -> Self {
        Self::new_error_with_message(
            interner,
            I::Span::dummy(),
            "ty::ConstKind::Error constructed but no error reported",
        )
    }

    #[track_caller]
    pub fn new_error_with_message(interner: I, span: I::Span, msg: impl ToString) -> Self {
        let reported = interner.span_delayed_bug(span, msg);
        Self::new_error(interner, reported)
    }

    pub fn is_trivially_wf(self) -> bool {
        match self.kind() {
            ConstKind::Param(_) | ConstKind::Placeholder(_) | ConstKind::Bound(..) => true,
            ConstKind::Infer(_)
            | ConstKind::Alias(..)
            | ConstKind::Value(_)
            | ConstKind::Error(_)
            | ConstKind::Expr(_) => false,
        }
    }

    pub fn is_ct_infer(self) -> bool {
        matches!(self.kind(), ConstKind::Infer(_))
    }

    pub fn ct_vid(self) -> Option<ConstVid> {
        match self.kind() {
            ConstKind::Infer(InferConst::Var(vid)) => Some(vid),
            _ => None,
        }
    }
}

impl<I: Interner> Flags for Const<I> {
    fn flags(&self) -> TypeFlags {
        self.0.get().flags
    }

    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.0.get().outer_exclusive_binder
    }
}

impl<I: Interner> IntoKind for Const<I> {
    type Kind = ConstKind<I>;

    fn kind(self) -> Self::Kind {
        *self.0.get()
    }
}

impl<I: Interner> TypeFoldable<I> for Const<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_const(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_const(self)
    }
}

impl<I: Interner> TypeVisitable<I> for Const<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_const(*self)
    }
}

impl<I: Interner> TypeSuperFoldable<I> for Const<I> {
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match self.kind() {
            ConstKind::Alias(is_rigid, alias_const) => {
                ConstKind::Alias(is_rigid, alias_const.try_fold_with(folder)?)
            }
            ConstKind::Value(v) => ConstKind::Value(v.try_fold_with(folder)?),
            ConstKind::Expr(e) => ConstKind::Expr(e.try_fold_with(folder)?),

            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(_)
            | ConstKind::Error(_) => return Ok(self),
        };
        if kind != self.kind() { Ok(Self::new(folder.cx(), kind)) } else { Ok(self) }
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        let kind = match self.kind() {
            ConstKind::Alias(is_rigid, alias_const) => {
                ConstKind::Alias(is_rigid, alias_const.fold_with(folder))
            }
            ConstKind::Value(v) => ConstKind::Value(v.fold_with(folder)),
            ConstKind::Expr(e) => ConstKind::Expr(e.fold_with(folder)),

            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(_)
            | ConstKind::Error(_) => return self,
        };
        if kind != self.kind() { Self::new(folder.cx(), kind) } else { self }
    }
}

impl<I: Interner> TypeSuperVisitable<I> for Const<I> {
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match self.kind() {
            ConstKind::Alias(_, alias_const) => alias_const.visit_with(visitor),
            ConstKind::Value(v) => v.visit_with(visitor),
            ConstKind::Expr(e) => e.visit_with(visitor),
            ConstKind::Error(e) => e.visit_with(visitor),

            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(_) => V::Result::output(),
        }
    }
}

impl<I: Interner> Relate<I> for Const<I> {
    fn relate<R: TypeRelation<I>>(relation: &mut R, a: Self, b: Self) -> RelateResult<I, Self> {
        relation.consts(a, b)
    }
}
