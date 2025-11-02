//! Things related to consts in the next-trait-solver.

use std::hash::Hash;

use hir_def::ConstParamId;
use macros::{TypeFoldable, TypeVisitable};
use rustc_ast_ir::visit::VisitorResult;
use rustc_type_ir::{
    BoundVar, BoundVarIndexKind, ConstVid, DebruijnIndex, FlagComputation, Flags, InferConst,
    TypeFoldable, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    WithCachedTypeInfo,
    inherent::{IntoKind, ParamEnv as _, PlaceholderLike, SliceLike},
    relate::Relate,
};

use crate::{
    MemoryMap,
    next_solver::{ClauseKind, ParamEnv, interner::InternedWrapperNoDebug},
};

use super::{BoundVarKind, DbInterner, ErrorGuaranteed, GenericArgs, Placeholder, Ty};

pub type ConstKind<'db> = rustc_type_ir::ConstKind<DbInterner<'db>>;
pub type UnevaluatedConst<'db> = rustc_type_ir::UnevaluatedConst<DbInterner<'db>>;

#[salsa::interned(constructor = new_)]
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
        crate::with_attached_db(|db| {
            let inner = &self.kind_(db).0;
            // SAFETY: The caller already has access to a `Const<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
    }

    pub fn error(interner: DbInterner<'db>) -> Self {
        Const::new(interner, ConstKind::Error(ErrorGuaranteed))
    }

    pub fn new_param(interner: DbInterner<'db>, param: ParamConst) -> Self {
        Const::new(interner, ConstKind::Param(param))
    }

    pub fn new_placeholder(interner: DbInterner<'db>, placeholder: PlaceholderConst) -> Self {
        Const::new(interner, ConstKind::Placeholder(placeholder))
    }

    pub fn new_bound(interner: DbInterner<'db>, index: DebruijnIndex, bound: BoundConst) -> Self {
        Const::new(interner, ConstKind::Bound(BoundVarIndexKind::Bound(index), bound))
    }

    pub fn new_valtree(
        interner: DbInterner<'db>,
        ty: Ty<'db>,
        memory: Box<[u8]>,
        memory_map: MemoryMap<'db>,
    ) -> Self {
        Const::new(
            interner,
            ConstKind::Value(ValueConst {
                ty,
                value: Valtree::new(ConstBytes { memory, memory_map }),
            }),
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

impl<'db> std::fmt::Debug for InternedWrapperNoDebug<WithCachedTypeInfo<ConstKind<'db>>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.internee.fmt(f)
    }
}

pub type PlaceholderConst = Placeholder<BoundConst>;

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

impl ParamConst {
    pub fn find_const_ty_from_env<'db>(self, env: ParamEnv<'db>) -> Ty<'db> {
        let mut candidates = env.caller_bounds().iter().filter_map(|clause| {
            // `ConstArgHasType` are never desugared to be higher ranked.
            match clause.kind().skip_binder() {
                ClauseKind::ConstArgHasType(param_ct, ty) => {
                    assert!(!(param_ct, ty).has_escaping_bound_vars());

                    match param_ct.kind() {
                        ConstKind::Param(param_ct) if param_ct.index == self.index => Some(ty),
                        _ => None,
                    }
                }
                _ => None,
            }
        });

        // N.B. it may be tempting to fix ICEs by making this function return
        // `Option<Ty<'db>>` instead of `Ty<'db>`; however, this is generally
        // considered to be a bandaid solution, since it hides more important
        // underlying issues with how we construct generics and predicates of
        // items. It's advised to fix the underlying issue rather than trying
        // to modify this function.
        let ty = candidates.next().unwrap_or_else(|| {
            panic!("cannot find `{self:?}` in param-env: {env:#?}");
        });
        assert!(
            candidates.next().is_none(),
            "did not expect duplicate `ConstParamHasTy` for `{self:?}` in param-env: {env:#?}"
        );
        ty
    }
}

/// A type-level constant value.
///
/// Represents a typed, fully evaluated constant.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, TypeFoldable, TypeVisitable)]
pub struct ValueConst<'db> {
    pub ty: Ty<'db>,
    // FIXME: Should we ignore this for TypeVisitable, TypeFoldable?
    #[type_visitable(ignore)]
    #[type_foldable(identity)]
    pub value: Valtree<'db>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstBytes<'db> {
    pub memory: Box<[u8]>,
    pub memory_map: MemoryMap<'db>,
}

impl Hash for ConstBytes<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.memory.hash(state)
    }
}

#[salsa::interned(constructor = new_, debug)]
pub struct Valtree<'db> {
    #[returns(ref)]
    bytes_: ConstBytes<'db>,
}

impl<'db> Valtree<'db> {
    pub fn new(bytes: ConstBytes<'db>) -> Self {
        crate::with_attached_db(|db| unsafe {
            // SAFETY: ¯\_(ツ)_/¯
            std::mem::transmute(Valtree::new_(db, bytes))
        })
    }

    pub fn inner(&self) -> &ConstBytes<'db> {
        crate::with_attached_db(|db| {
            let inner = self.bytes_(db);
            // SAFETY: The caller already has access to a `Valtree<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, TypeVisitable, TypeFoldable)]
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
    fn new_infer(interner: DbInterner<'db>, var: InferConst) -> Self {
        Const::new(interner, ConstKind::Infer(var))
    }

    fn new_var(interner: DbInterner<'db>, var: ConstVid) -> Self {
        Const::new(interner, ConstKind::Infer(InferConst::Var(var)))
    }

    fn new_bound(interner: DbInterner<'db>, debruijn: DebruijnIndex, var: BoundConst) -> Self {
        Const::new(interner, ConstKind::Bound(BoundVarIndexKind::Bound(debruijn), var))
    }

    fn new_anon_bound(interner: DbInterner<'db>, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Const::new(
            interner,
            ConstKind::Bound(BoundVarIndexKind::Bound(debruijn), BoundConst { var }),
        )
    }

    fn new_canonical_bound(interner: DbInterner<'db>, var: BoundVar) -> Self {
        Const::new(interner, ConstKind::Bound(BoundVarIndexKind::Canonical, BoundConst { var }))
    }

    fn new_placeholder(
        interner: DbInterner<'db>,
        param: <DbInterner<'db> as rustc_type_ir::Interner>::PlaceholderConst,
    ) -> Self {
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

    fn new_error(interner: DbInterner<'db>, guar: ErrorGuaranteed) -> Self {
        Const::new(interner, ConstKind::Error(guar))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundConst {
    pub var: BoundVar,
}

impl<'db> rustc_type_ir::inherent::BoundVarLike<DbInterner<'db>> for BoundConst {
    fn var(self) -> BoundVar {
        self.var
    }

    fn assert_eq(self, var: BoundVarKind) {
        var.expect_const()
    }
}

impl<'db> PlaceholderLike<DbInterner<'db>> for PlaceholderConst {
    type Bound = BoundConst;

    fn universe(self) -> rustc_type_ir::UniverseIndex {
        self.universe
    }

    fn var(self) -> rustc_type_ir::BoundVar {
        self.bound.var
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        Placeholder { universe: ui, bound: self.bound }
    }

    fn new(ui: rustc_type_ir::UniverseIndex, var: BoundConst) -> Self {
        Placeholder { universe: ui, bound: var }
    }
    fn new_anon(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        Placeholder { universe: ui, bound: BoundConst { var } }
    }
}

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
