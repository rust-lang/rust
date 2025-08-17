//! Things related to consts in the next-trait-solver.

use std::hash::Hash;

use intern::{Interned, Symbol};
use rustc_ast_ir::try_visit;
use rustc_ast_ir::visit::VisitorResult;
use rustc_type_ir::{
    BoundVar, FlagComputation, Flags, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable,
    TypeVisitable, WithCachedTypeInfo,
    inherent::{IntoKind, PlaceholderLike},
    relate::Relate,
};

use crate::{ConstScalar, MemoryMap, interner::InternedWrapperNoDebug};

use super::{BoundVarKind, DbInterner, ErrorGuaranteed, GenericArgs, Placeholder, Ty};

pub type ConstKind<'db> = rustc_type_ir::ConstKind<DbInterner<'db>>;
pub type UnevaluatedConst<'db> = rustc_type_ir::UnevaluatedConst<DbInterner<'db>>;

#[salsa::interned(constructor = new_, debug)]
pub struct Const<'db> {
    #[returns(ref)]
    kind_: InternedWrapperNoDebug<WithCachedTypeInfo<ConstKind<'db>>>,
}

impl<'db> Const<'db> {
    pub fn new(interner: DbInterner<'db>, kind: ConstKind<'db>) -> Self {
        let flags = FlagComputation::for_const_kind(&kind);
        let cached = WithCachedTypeInfo {
            internee: kind,
            flags: flags.flags,
            outer_exclusive_binder: flags.outer_exclusive_binder,
        };
        Const::new_(interner.db(), InternedWrapperNoDebug(cached))
    }

    pub fn inner(&self) -> &WithCachedTypeInfo<ConstKind<'db>> {
        salsa::with_attached_database(|db| {
            let inner = &self.kind_(db).0;
            // SAFETY: The caller already has access to a `Const<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
        .unwrap()
    }

    pub fn error(interner: DbInterner<'db>) -> Self {
        Const::new(interner, rustc_type_ir::ConstKind::Error(ErrorGuaranteed))
    }

    pub fn new_param(interner: DbInterner<'db>, param: ParamConst) -> Self {
        Const::new(interner, rustc_type_ir::ConstKind::Param(param))
    }

    pub fn new_placeholder(interner: DbInterner<'db>, placeholder: PlaceholderConst) -> Self {
        Const::new(interner, ConstKind::Placeholder(placeholder))
    }

    pub fn is_ct_infer(&self) -> bool {
        matches!(&self.inner().internee, ConstKind::Infer(_))
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

impl<'db> std::fmt::Debug for InternedWrapperNoDebug<WithCachedTypeInfo<ConstKind<'db>>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.internee.fmt(f)
    }
}

pub type PlaceholderConst = Placeholder<rustc_type_ir::BoundVar>;

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct ParamConst {
    pub index: u32,
}

impl std::fmt::Debug for ParamConst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
    }
}

/// A type-level constant value.
///
/// Represents a typed, fully evaluated constant.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ValueConst<'db> {
    pub(crate) ty: Ty<'db>,
    pub(crate) value: Valtree<'db>,
}

impl<'db> ValueConst<'db> {
    pub fn new(ty: Ty<'db>, bytes: ConstBytes<'db>) -> Self {
        let value = Valtree::new(bytes);
        ValueConst { ty, value }
    }
}

impl<'db> rustc_type_ir::inherent::ValueConst<DbInterner<'db>> for ValueConst<'db> {
    fn ty(self) -> Ty<'db> {
        self.ty
    }

    fn valtree(self) -> Valtree<'db> {
        self.value
    }
}

impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for ValueConst<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        self.ty.visit_with(visitor)
    }
}

impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for ValueConst<'db> {
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        ValueConst { ty: self.ty.fold_with(folder), value: self.value }
    }
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(ValueConst { ty: self.ty.try_fold_with(folder)?, value: self.value })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstBytes<'db>(pub Box<[u8]>, pub MemoryMap<'db>);

impl Hash for ConstBytes<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[salsa::interned(constructor = new_, debug)]
pub struct Valtree<'db> {
    #[returns(ref)]
    bytes_: ConstBytes<'db>,
}

impl<'db> Valtree<'db> {
    pub fn new(bytes: ConstBytes<'db>) -> Self {
        salsa::with_attached_database(|db| unsafe {
            // SAFETY: ¯\_(ツ)_/¯
            std::mem::transmute(Valtree::new_(db, bytes))
        })
        .unwrap()
    }

    pub fn inner(&self) -> &ConstBytes<'db> {
        salsa::with_attached_database(|db| {
            let inner = self.bytes_(db);
            // SAFETY: The caller already has access to a `Valtree<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
        .unwrap()
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
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
    fn new_infer(interner: DbInterner<'db>, var: rustc_type_ir::InferConst) -> Self {
        Const::new(interner, ConstKind::Infer(var))
    }

    fn new_var(interner: DbInterner<'db>, var: rustc_type_ir::ConstVid) -> Self {
        Const::new(interner, ConstKind::Infer(rustc_type_ir::InferConst::Var(var)))
    }

    fn new_bound(
        interner: DbInterner<'db>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: BoundVar,
    ) -> Self {
        Const::new(interner, ConstKind::Bound(debruijn, var))
    }

    fn new_anon_bound(
        interner: DbInterner<'db>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        Const::new(interner, ConstKind::Bound(debruijn, var))
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

    fn new_error(interner: DbInterner<'db>, guar: ErrorGuaranteed) -> Self {
        Const::new(interner, ConstKind::Error(guar))
    }

    fn new_placeholder(
        interner: DbInterner<'db>,
        param: <DbInterner<'db> as rustc_type_ir::Interner>::PlaceholderConst,
    ) -> Self {
        Const::new(interner, ConstKind::Placeholder(param))
    }
}

impl<'db> PlaceholderLike<DbInterner<'db>> for PlaceholderConst {
    type Bound = rustc_type_ir::BoundVar;

    fn universe(self) -> rustc_type_ir::UniverseIndex {
        self.universe
    }

    fn var(self) -> rustc_type_ir::BoundVar {
        self.bound
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        Placeholder { universe: ui, bound: self.bound }
    }

    fn new(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        Placeholder { universe: ui, bound: var }
    }
    fn new_anon(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        Placeholder { universe: ui, bound: var }
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for ExprConst {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        // Ensure we get back to this when we fill in the fields
        let ExprConst = &self;
        V::Result::output()
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for ExprConst {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(ExprConst)
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        ExprConst
    }
}

impl<'db> Relate<DbInterner<'db>> for ExprConst {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
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
