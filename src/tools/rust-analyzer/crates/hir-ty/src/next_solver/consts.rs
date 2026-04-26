//! Things related to consts in the next-trait-solver.

mod valtree;

use std::hash::Hash;

use hir_def::ConstParamId;
use intern::{Interned, InternedRef, impl_internable};
use macros::{GenericTypeVisitable, TypeFoldable, TypeVisitable};
use rustc_ast_ir::visit::VisitorResult;
use rustc_type_ir::{
    BoundVar, BoundVarIndexKind, ConstVid, DebruijnIndex, FlagComputation, Flags,
    GenericTypeVisitable, InferConst, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable,
    TypeVisitable, WithCachedTypeInfo, inherent::IntoKind, relate::Relate,
};

use crate::{
    ParamEnvAndCrate,
    next_solver::{
        AllocationData, impl_foldable_for_interned_slice, impl_stored_interned, interned_slice,
    },
};

use super::{DbInterner, ErrorGuaranteed, GenericArgs, Ty};

pub use self::valtree::*;

pub type ConstKind<'db> = rustc_type_ir::ConstKind<DbInterner<'db>>;
pub type UnevaluatedConst<'db> = rustc_type_ir::UnevaluatedConst<DbInterner<'db>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Const<'db> {
    pub(super) interned: InternedRef<'db, ConstInterned>,
}

#[derive(PartialEq, Eq, Hash, GenericTypeVisitable)]
#[repr(align(4))] // Required for `GenericArg` bit-tagging.
pub(super) struct ConstInterned(pub(super) WithCachedTypeInfo<ConstKind<'static>>);

impl_internable!(gc; ConstInterned);
impl_stored_interned!(ConstInterned, Const, StoredConst);

const _: () = {
    const fn is_copy<T: Copy>() {}
    is_copy::<Const<'static>>();
};

impl<'db> Const<'db> {
    pub fn new(_interner: DbInterner<'db>, kind: ConstKind<'db>) -> Self {
        let kind = unsafe { std::mem::transmute::<ConstKind<'db>, ConstKind<'static>>(kind) };
        let flags = FlagComputation::for_const_kind(&kind);
        let cached = WithCachedTypeInfo {
            internee: kind,
            flags: flags.flags,
            outer_exclusive_binder: flags.outer_exclusive_binder,
        };
        Self { interned: Interned::new_gc(ConstInterned(cached)) }
    }

    pub fn inner(&self) -> &WithCachedTypeInfo<ConstKind<'db>> {
        let inner = &self.interned.0;
        unsafe {
            std::mem::transmute::<
                &WithCachedTypeInfo<ConstKind<'static>>,
                &WithCachedTypeInfo<ConstKind<'db>>,
            >(inner)
        }
    }

    pub fn error(interner: DbInterner<'db>) -> Self {
        interner.default_types().consts.error
    }

    pub fn new_param(interner: DbInterner<'db>, param: ParamConst) -> Self {
        Const::new(interner, ConstKind::Param(param))
    }

    pub fn new_placeholder(interner: DbInterner<'db>, placeholder: PlaceholderConst<'db>) -> Self {
        Const::new(interner, ConstKind::Placeholder(placeholder))
    }

    pub fn new_bound(
        interner: DbInterner<'db>,
        index: DebruijnIndex,
        bound: BoundConst<'db>,
    ) -> Self {
        Const::new(interner, ConstKind::Bound(BoundVarIndexKind::Bound(index), bound))
    }

    pub fn new_valtree(interner: DbInterner<'db>, ty: Ty<'db>, kind: ValTreeKind<'db>) -> Self {
        Const::new(interner, ConstKind::Value(ValueConst { ty, value: ValTree::new(kind) }))
    }

    pub fn new_from_allocation(
        interner: DbInterner<'db>,
        allocation: &AllocationData<'db>,
        param_env: ParamEnvAndCrate<'db>,
    ) -> Self {
        allocation_to_const(
            interner,
            allocation.ty,
            &allocation.memory,
            &allocation.memory_map,
            param_env,
        )
    }

    pub fn is_ct_infer(&self) -> bool {
        matches!(self.kind(), ConstKind::Infer(_))
    }

    pub fn is_error(&self) -> bool {
        matches!(self.kind(), ConstKind::Error(_))
    }

    pub fn is_trivially_wf(self) -> bool {
        match self.kind() {
            ConstKind::Param(_) | ConstKind::Placeholder(_) | ConstKind::Bound(..) => true,
            ConstKind::Infer(_)
            | ConstKind::Unevaluated(..)
            | ConstKind::Value(_)
            | ConstKind::Error(_)
            | ConstKind::Expr(_) => false,
        }
    }
}

impl<'db> std::fmt::Debug for Const<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().internee.fmt(f)
    }
}

pub type PlaceholderConst<'db> = rustc_type_ir::PlaceholderConst<DbInterner<'db>>;

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct ParamConst {
    // FIXME: See `ParamTy`.
    pub id: ConstParamId,
    pub index: u32,
}

impl std::fmt::Debug for ParamConst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
    }
}

#[derive(
    Copy, Clone, Debug, Hash, PartialEq, Eq, TypeVisitable, TypeFoldable, GenericTypeVisitable,
)]
pub struct ExprConst;

impl rustc_type_ir::inherent::ParamLike for ParamConst {
    fn index(self) -> u32 {
        self.index
    }
}

impl<'db> IntoKind for Const<'db> {
    type Kind = ConstKind<'db>;

    fn kind(self) -> Self::Kind {
        self.inner().internee
    }
}

impl<'db, V: super::WorldExposer> GenericTypeVisitable<V> for Const<'db> {
    fn generic_visit_with(&self, visitor: &mut V) {
        if visitor.on_interned(self.interned).is_continue() {
            self.kind().generic_visit_with(visitor);
        }
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for Const<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        visitor.visit_const(*self)
    }
}

impl<'db> TypeSuperVisitable<DbInterner<'db>> for Const<'db> {
    fn super_visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        match self.kind() {
            ConstKind::Unevaluated(uv) => uv.visit_with(visitor),
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

impl<'db> TypeFoldable<DbInterner<'db>> for Const<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_const(self)
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        folder.fold_const(self)
    }
}

impl<'db> TypeSuperFoldable<DbInterner<'db>> for Const<'db> {
    fn try_super_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match self.kind() {
            ConstKind::Unevaluated(uv) => ConstKind::Unevaluated(uv.try_fold_with(folder)?),
            ConstKind::Value(v) => ConstKind::Value(v.try_fold_with(folder)?),
            ConstKind::Expr(e) => ConstKind::Expr(e.try_fold_with(folder)?),

            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(_)
            | ConstKind::Error(_) => return Ok(self),
        };
        if kind != self.kind() { Ok(Const::new(folder.cx(), kind)) } else { Ok(self) }
    }
    fn super_fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Self {
        let kind = match self.kind() {
            ConstKind::Unevaluated(uv) => ConstKind::Unevaluated(uv.fold_with(folder)),
            ConstKind::Value(v) => ConstKind::Value(v.fold_with(folder)),
            ConstKind::Expr(e) => ConstKind::Expr(e.fold_with(folder)),

            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(_)
            | ConstKind::Error(_) => return self,
        };
        if kind != self.kind() { Const::new(folder.cx(), kind) } else { self }
    }
}

impl<'db> Relate<DbInterner<'db>> for Const<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        relation.consts(a, b)
    }
}

impl<'db> Flags for Const<'db> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        self.inner().flags
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        self.inner().outer_exclusive_binder
    }
}

impl<'db> rustc_type_ir::inherent::Const<DbInterner<'db>> for Const<'db> {
    fn new_infer(interner: DbInterner<'db>, var: InferConst) -> Self {
        Const::new(interner, ConstKind::Infer(var))
    }

    fn new_var(interner: DbInterner<'db>, var: ConstVid) -> Self {
        Const::new(interner, ConstKind::Infer(InferConst::Var(var)))
    }

    fn new_bound(interner: DbInterner<'db>, debruijn: DebruijnIndex, var: BoundConst<'db>) -> Self {
        Const::new(interner, ConstKind::Bound(BoundVarIndexKind::Bound(debruijn), var))
    }

    fn new_anon_bound(interner: DbInterner<'db>, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Const::new(
            interner,
            ConstKind::Bound(BoundVarIndexKind::Bound(debruijn), BoundConst::new(var)),
        )
    }

    fn new_canonical_bound(interner: DbInterner<'db>, var: BoundVar) -> Self {
        Const::new(interner, ConstKind::Bound(BoundVarIndexKind::Canonical, BoundConst::new(var)))
    }

    fn new_placeholder(interner: DbInterner<'db>, param: PlaceholderConst<'db>) -> Self {
        Const::new(interner, ConstKind::Placeholder(param))
    }

    fn new_unevaluated(
        interner: DbInterner<'db>,
        uv: rustc_type_ir::UnevaluatedConst<DbInterner<'db>>,
    ) -> Self {
        Const::new(interner, ConstKind::Unevaluated(uv))
    }

    fn new_expr(interner: DbInterner<'db>, expr: ExprConst) -> Self {
        Const::new(interner, ConstKind::Expr(expr))
    }

    fn new_error(interner: DbInterner<'db>, _guar: ErrorGuaranteed) -> Self {
        Const::error(interner)
    }
}

pub type BoundConst<'db> = rustc_type_ir::BoundConst<DbInterner<'db>>;

impl<'db> Relate<DbInterner<'db>> for ExprConst {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        _relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        // Ensure we get back to this when we fill in the fields
        let ExprConst = b;
        Ok(a)
    }
}

impl<'db> rustc_type_ir::inherent::ExprConst<DbInterner<'db>> for ExprConst {
    fn args(self) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        // Ensure we get back to this when we fill in the fields
        let ExprConst = self;
        GenericArgs::default()
    }
}

interned_slice!(ConstsStorage, Consts, StoredConsts, consts, Const<'db>, Const<'static>);
impl_foldable_for_interned_slice!(Consts);
