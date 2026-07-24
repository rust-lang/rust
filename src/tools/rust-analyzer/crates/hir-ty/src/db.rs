//! The home of `HirDatabase`, which is the Salsa database containing all the
//! type inference-related queries.

use arrayvec::ArrayVec;
use base_db::{Crate, SourceDatabase, target::TargetLoadError};
use either::Either;
use hir_def::{
    AdtId, BuiltinDeriveImplId, CallableDefId, ConstId, ConstParamId, EnumVariantId,
    ExpressionStoreOwnerId, FunctionId, GenericDefId, HasModule, ImplId, LocalFieldId, ModuleId,
    StaticId, TraitId, TypeAliasId, VariantId,
    builtin_derive::BuiltinDeriveImplMethod,
    expr_store::ExpressionStore,
    hir::{ClosureKind, ExprId, generics::LocalTypeOrConstParamId},
    layout::TargetDataLayout,
    resolver::{HasResolver, Resolver},
    signatures::{ConstSignature, StaticSignature},
};
use la_arena::ArenaMap;
use salsa::Update;
use span::Edition;
use stdx::impl_from;
use triomphe::Arc;

use crate::{
    FieldType, GenericDefaultsRef, GenericPredicates, ImplTraitId, InferBodyId, TyDefId,
    TyLoweringResult, ValueTyDefId,
    consteval::ConstEvalError,
    dyn_compatibility::DynCompatibilityViolation,
    layout::{Layout, LayoutError},
    lower::{GenericDefaults, TrackedStructToken, TypeAliasBounds},
    mir::{MirBody, MirLowerError},
    next_solver::{
        Allocation, Clause, EarlyBinder, GenericArgs, ParamEnv, PolyFnSig, StoredClauses,
        StoredEarlyBinder, StoredGenericArgs, StoredPolyFnSig, StoredTraitRef, StoredTy, TraitRef,
        Ty, VariancesOf,
    },
    traits::{ParamEnvAndCrate, StoredParamEnvAndCrate},
};

#[salsa::db]
pub trait HirDatabase: SourceDatabase + 'static {
    /// Manual implementation of upcasting from `dyn SourceDatabase` to `dyn HirDatabase`.
    ///
    /// This function is needed because Rust can't perform this upcasting automatically
    /// in the general case, as `Self` could be unsized.
    fn as_dyn(&self) -> &dyn HirDatabase;

    // region:mir

    // FIXME: Collapse `mir_body_for_closure` into `mir_body`
    // and `monomorphized_mir_body_for_closure` into `monomorphized_mir_body`
    fn mir_body<'db>(
        &'db self,
        def: InferBodyId<'db>,
    ) -> Result<&'db MirBody<'db>, MirLowerError<'db>> {
        let db = self.as_dyn();
        crate::mir::mir_body_query(db, def).map_err(|err| err.clone())
    }

    fn mir_body_for_closure<'db>(
        &'db self,
        def: InternedClosureId<'db>,
    ) -> Result<&'db MirBody<'db>, MirLowerError<'db>> {
        let db = self.as_dyn();
        crate::mir::mir_body_for_closure_query(db, def).map_err(|err| err.clone())
    }

    fn monomorphized_mir_body<'db>(
        &'db self,
        def: InferBodyId<'db>,
        subst: StoredGenericArgs,
        env: StoredParamEnvAndCrate,
    ) -> Result<&'db MirBody<'db>, MirLowerError<'db>> {
        let db = self.as_dyn();
        crate::mir::monomorphized_mir_body_query(db, def, subst, env).map_err(|err| err.clone())
    }

    fn monomorphized_mir_body_for_closure<'db>(
        &'db self,
        def: InternedClosureId<'db>,
        subst: StoredGenericArgs,
        env: StoredParamEnvAndCrate,
    ) -> Result<&'db MirBody<'db>, MirLowerError<'db>> {
        let db = self.as_dyn();
        crate::mir::monomorphized_mir_body_for_closure_query(db, def, subst, env)
            .map_err(|err| err.clone())
    }

    fn const_eval<'db>(
        &'db self,
        def: ConstId,
        subst: GenericArgs<'db>,
        trait_env: Option<ParamEnvAndCrate<'db>>,
    ) -> Result<Allocation<'db>, ConstEvalError<'db>> {
        let db = self.as_dyn();
        crate::consteval::const_eval(db, def, subst, trait_env)
    }

    fn anon_const_eval<'db>(
        &'db self,
        def: AnonConstId<'db>,
        subst: GenericArgs<'db>,
        trait_env: Option<ParamEnvAndCrate<'db>>,
    ) -> Result<Allocation<'db>, ConstEvalError<'db>> {
        let db = self.as_dyn();
        crate::consteval::anon_const_eval(db, def, subst, trait_env)
    }

    fn const_eval_static<'db>(
        &'db self,
        def: StaticId,
    ) -> Result<Allocation<'db>, ConstEvalError<'db>> {
        let db = self.as_dyn();
        crate::consteval::const_eval_static(db, def)
    }

    fn const_eval_discriminant<'db>(
        &'db self,
        def: EnumVariantId,
    ) -> Result<i128, ConstEvalError<'db>> {
        let db = self.as_dyn();
        crate::consteval::const_eval_discriminant_variant(db, def)
    }

    fn lookup_impl_method<'db>(
        &'db self,
        env: ParamEnvAndCrate<'db>,
        func: FunctionId,
        fn_subst: GenericArgs<'db>,
    ) -> (Either<FunctionId, (BuiltinDeriveImplId, BuiltinDeriveImplMethod)>, GenericArgs<'db>)
    {
        let db = self.as_dyn();
        crate::method_resolution::lookup_impl_method_query(db, env, func, fn_subst)
    }

    // endregion:mir

    fn layout_of_adt(
        &self,
        def: AdtId,
        args: StoredGenericArgs,
        trait_env: StoredParamEnvAndCrate,
    ) -> Result<Arc<Layout>, LayoutError> {
        let db = self.as_dyn();
        crate::layout::layout_of_adt_query(db, def, args, trait_env)
    }

    fn layout_of_ty(
        &self,
        ty: StoredTy,
        env: StoredParamEnvAndCrate,
    ) -> Result<Arc<Layout>, LayoutError> {
        let db = self.as_dyn();
        crate::layout::layout_of_ty_query(db, ty, env)
    }

    fn target_data_layout(&self, krate: Crate) -> Result<&TargetDataLayout, TargetLoadError> {
        let db = self.as_dyn();
        crate::layout::target_data_layout_query(db, krate).map_err(|err| err.clone())
    }

    fn dyn_compatibility_of_trait(&self, trait_: TraitId) -> Option<DynCompatibilityViolation> {
        let db = self.as_dyn();
        crate::dyn_compatibility::dyn_compatibility_of_trait_query(db, trait_)
    }

    fn ty<'db>(&'db self, def: TyDefId) -> EarlyBinder<'db, Ty<'db>> {
        let db = self.as_dyn();
        crate::lower::ty_query(db, def)
    }

    fn type_for_type_alias_with_diagnostics<'db>(
        &'db self,
        def: TypeAliasId,
    ) -> &'db TyLoweringResult<'db, StoredEarlyBinder<StoredTy>> {
        let db = self.as_dyn();
        crate::lower::type_for_type_alias_with_diagnostics(db, def)
    }

    /// Returns the type of the value of the given constant, or `None` if the `ValueTyDefId` is
    /// a `StructId` or `EnumVariantId` with a record constructor.
    fn value_ty<'db>(&'db self, def: ValueTyDefId) -> Option<EarlyBinder<'db, Ty<'db>>> {
        let db = self.as_dyn();
        crate::lower::value_ty(db, def)
    }

    fn type_for_const<'db>(&'db self, def: ConstId) -> EarlyBinder<'db, Ty<'db>> {
        let db = self.as_dyn();
        crate::lower::type_for_const(db, def)
    }

    fn type_for_const_with_diagnostics<'db>(
        &'db self,
        def: ConstId,
    ) -> &'db TyLoweringResult<'db, StoredEarlyBinder<StoredTy>> {
        let db = self.as_dyn();
        crate::lower::type_for_const_with_diagnostics(db, def)
    }

    fn type_for_static<'db>(&'db self, def: StaticId) -> EarlyBinder<'db, Ty<'db>> {
        let db = self.as_dyn();
        crate::lower::type_for_static(db, def)
    }

    fn type_for_static_with_diagnostics<'db>(
        &'db self,
        def: StaticId,
    ) -> &'db TyLoweringResult<'db, StoredEarlyBinder<StoredTy>> {
        let db = self.as_dyn();
        crate::lower::type_for_static_with_diagnostics(db, def)
    }

    fn impl_self_ty_with_diagnostics<'db>(
        &'db self,
        def: ImplId,
    ) -> &'db TyLoweringResult<'db, StoredEarlyBinder<StoredTy>> {
        let db = self.as_dyn();
        crate::lower::impl_self_ty_with_diagnostics(db, def)
    }

    fn impl_self_ty<'db>(&'db self, def: ImplId) -> EarlyBinder<'db, Ty<'db>> {
        let db = self.as_dyn();
        crate::lower::impl_self_ty_query(db, def)
    }

    fn const_param_types_with_diagnostics<'db>(
        &'db self,
        def: GenericDefId,
    ) -> &'db TyLoweringResult<'db, ArenaMap<LocalTypeOrConstParamId, StoredTy>> {
        let db = self.as_dyn();
        crate::lower::const_param_types_with_diagnostics(db, def)
    }

    fn const_param_types(&self, def: GenericDefId) -> &ArenaMap<LocalTypeOrConstParamId, StoredTy> {
        let db = self.as_dyn();
        crate::lower::const_param_types(db, def)
    }

    fn const_param_ty<'db>(&'db self, def: ConstParamId) -> Ty<'db> {
        let db = self.as_dyn();
        crate::lower::const_param_ty(db, def)
    }

    fn impl_trait_with_diagnostics<'db>(
        &'db self,
        def: ImplId,
    ) -> &'db Option<TyLoweringResult<'db, StoredEarlyBinder<StoredTraitRef>>> {
        let db = self.as_dyn();
        crate::lower::impl_trait_with_diagnostics(db, def)
    }

    fn impl_trait<'db>(&'db self, def: ImplId) -> Option<EarlyBinder<'db, TraitRef<'db>>> {
        let db = self.as_dyn();
        crate::lower::impl_trait_query(db, def)
    }

    fn field_types_with_diagnostics<'db>(
        &'db self,
        var: VariantId,
    ) -> &'db TyLoweringResult<'db, ArenaMap<LocalFieldId, FieldType>> {
        let db = self.as_dyn();
        crate::lower::field_types_with_diagnostics(db, var)
    }

    fn field_types(&self, var: VariantId) -> &ArenaMap<LocalFieldId, FieldType> {
        let db = self.as_dyn();
        crate::lower::field_types_query(db, var)
    }

    fn callable_item_signature<'db>(
        &'db self,
        def: CallableDefId,
    ) -> EarlyBinder<'db, PolyFnSig<'db>> {
        let db = self.as_dyn();
        crate::lower::callable_item_signature(db, def)
    }

    fn callable_item_signature_with_diagnostics<'db>(
        &'db self,
        def: CallableDefId,
    ) -> &'db TyLoweringResult<'db, StoredEarlyBinder<StoredPolyFnSig>> {
        let db = self.as_dyn();
        crate::lower::callable_item_signature_with_diagnostics(db, def)
    }

    fn trait_environment<'db>(&'db self, def: GenericDefId) -> ParamEnv<'db> {
        let db = self.as_dyn();
        crate::lower::trait_environment(db, def)
    }

    fn generic_defaults_with_diagnostics<'db>(
        &'db self,
        def: GenericDefId,
    ) -> &'db TyLoweringResult<'db, GenericDefaults> {
        let db = self.as_dyn();
        crate::lower::generic_defaults_with_diagnostics(db, def)
    }

    /// This returns an empty list if no parameter has default.
    ///
    /// The binders of the returned defaults are only up to (not including) this parameter.
    fn generic_defaults(&self, def: GenericDefId) -> GenericDefaultsRef<'_> {
        let db = self.as_dyn();
        crate::lower::generic_defaults(db, def)
    }

    fn type_alias_bounds_with_diagnostics<'db>(
        &'db self,
        type_alias: TypeAliasId,
    ) -> &'db TyLoweringResult<'db, TypeAliasBounds<StoredEarlyBinder<StoredClauses>>> {
        let db = self.as_dyn();
        crate::lower::type_alias_bounds_with_diagnostics(db, type_alias)
    }

    fn type_alias_bounds<'db>(
        &'db self,
        type_alias: TypeAliasId,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        let db = self.as_dyn();
        crate::lower::type_alias_bounds(db, type_alias)
    }

    fn type_alias_self_bounds<'db>(
        &'db self,
        type_alias: TypeAliasId,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        let db = self.as_dyn();
        crate::lower::type_alias_self_bounds(db, type_alias)
    }

    fn variances_of<'db>(&'db self, def: GenericDefId) -> VariancesOf<'db> {
        let db = self.as_dyn();
        crate::variance::variances_of(db, def)
    }
}

#[salsa::db]
impl<T: SourceDatabase> HirDatabase for T {
    fn as_dyn(&self) -> &dyn HirDatabase {
        self
    }
}

#[test]
fn hir_database_is_dyn_compatible() {
    fn _assert_dyn_compatible(_: &dyn HirDatabase) {}
}

#[salsa::interned(debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedOpaqueTyId {
    pub loc: ImplTraitId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct InternedClosure<'db> {
    pub owner: InferBodyId<'db>,
    pub expr: ExprId,
    pub kind: ClosureKind,
}

#[salsa::interned(constructor = new_impl, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedClosureId<'db> {
    pub loc: InternedClosure<'db>,
}

impl<'db> InternedClosureId<'db> {
    #[inline]
    pub fn new(db: &'db dyn HirDatabase, loc: InternedClosure<'db>) -> Self {
        if cfg!(debug_assertions) {
            let store = ExpressionStore::of(db, loc.owner.expression_store_owner(db));
            let expr = &store[loc.expr];
            assert!(
                matches!(
                    expr,
                    hir_def::hir::Expr::Closure {
                        closure_kind: hir_def::hir::ClosureKind::Closure,
                        ..
                    }
                ),
                "expected a closure, found {expr:?}"
            );
        }

        Self::new_impl(db, loc)
    }
}

#[salsa::interned(constructor = new_impl, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedCoroutineId<'db> {
    pub loc: InternedClosure<'db>,
}

impl<'db> InternedCoroutineId<'db> {
    #[inline]
    pub fn new(db: &'db dyn HirDatabase, loc: InternedClosure<'db>) -> Self {
        if cfg!(debug_assertions) {
            let store = ExpressionStore::of(db, loc.owner.expression_store_owner(db));
            let expr = &store[loc.expr];
            assert!(
                matches!(
                    expr,
                    hir_def::hir::Expr::Closure {
                        closure_kind: hir_def::hir::ClosureKind::OldCoroutine(_)
                            | hir_def::hir::ClosureKind::Coroutine { .. },
                        ..
                    }
                ),
                "expected a coroutine, found {expr:?}"
            );
        }

        Self::new_impl(db, loc)
    }
}

#[salsa::interned(constructor = new_impl, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedCoroutineClosureId<'db> {
    pub loc: InternedClosure<'db>,
}

impl<'db> InternedCoroutineClosureId<'db> {
    #[inline]
    pub fn new(db: &'db dyn HirDatabase, loc: InternedClosure<'db>) -> Self {
        if cfg!(debug_assertions) {
            let store = ExpressionStore::of(db, loc.owner.expression_store_owner(db));
            let expr = &store[loc.expr];
            assert!(
                matches!(
                    expr,
                    hir_def::hir::Expr::Closure {
                        closure_kind: hir_def::hir::ClosureKind::CoroutineClosure(_),
                        ..
                    }
                ),
                "expected a coroutine closure, found {expr:?}"
            );
        }

        Self::new_impl(db, loc)
    }
}

/// An anonymous const expression that appears in a type position (e.g., array lengths,
/// const generic arguments like `{ N + 1 }`, or const param defaults). Unlike named constants,
/// these don't have their own `Body` — their expressions live in the parent's signature `ExpressionStore`.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct AnonConstLoc {
    /// The owner store containing this expression.
    pub owner: ExpressionStoreOwnerId,
    /// The ExprId within the owner's ExpressionStore that is the root
    /// of this anonymous const expression.
    pub expr: ExprId,
    pub ty: StoredEarlyBinder<StoredTy>,
    /// Whether to allow using generic params from the owner.
    /// true for array repeats, false for everything else.
    pub(crate) allow_using_generic_params: bool,
}

#[salsa::interned(debug, revisions = usize::MAX, constructor = new_)]
#[derive(PartialOrd, Ord)]
pub struct AnonConstId {
    #[returns(ref)]
    pub loc: AnonConstLoc,
}

impl<'db> AnonConstId<'db> {
    pub(crate) fn new(
        db: &'db dyn SourceDatabase,
        loc: AnonConstLoc,
        token: TrackedStructToken,
    ) -> Self {
        _ = token;
        AnonConstId::new_(db, loc)
    }
}

impl HasModule for AnonConstId<'_> {
    fn module(&self, db: &dyn SourceDatabase) -> ModuleId {
        self.loc(db).owner.module(db)
    }
}

impl HasResolver for AnonConstId<'_> {
    fn resolver(self, db: &dyn SourceDatabase) -> Resolver<'_> {
        self.loc(db).owner.resolver(db)
    }
}

impl<'db> AnonConstId<'db> {
    pub fn all_from_signature(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> ArrayVec<&'db [Self], 5> {
        let mut result = ArrayVec::new();

        // Queries common to all generic defs:
        result.push(db.generic_defaults_with_diagnostics(def).defined_anon_consts());
        result.push(GenericPredicates::query_with_diagnostics(db, def).defined_anon_consts());
        result.push(db.const_param_types_with_diagnostics(def).defined_anon_consts());

        match def {
            GenericDefId::ImplId(id) => {
                result.push(db.impl_self_ty_with_diagnostics(id).defined_anon_consts());
                if let Some(trait_ref) = db.impl_trait_with_diagnostics(id) {
                    result.push(trait_ref.defined_anon_consts());
                }
            }
            GenericDefId::TypeAliasId(id) => {
                result.push(db.type_for_type_alias_with_diagnostics(id).defined_anon_consts());
                result.push(db.type_alias_bounds_with_diagnostics(id).defined_anon_consts());
            }
            GenericDefId::FunctionId(id) => result
                .push(db.callable_item_signature_with_diagnostics(id.into()).defined_anon_consts()),
            GenericDefId::ConstId(def) => {
                result.push(db.type_for_const_with_diagnostics(def).defined_anon_consts())
            }
            GenericDefId::StaticId(def) => {
                result.push(db.type_for_static_with_diagnostics(def).defined_anon_consts())
            }
            GenericDefId::TraitId(_) | GenericDefId::AdtId(_) => {}
        }

        result
    }
}

/// A constant, which might appears as a const item, an anonymous const block in expressions
/// or patterns, or as a constant in types with const generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Supertype)]
pub enum GeneralConstId<'db> {
    ConstId(ConstId),
    StaticId(StaticId),
    AnonConstId(AnonConstId<'db>),
}

impl_from!(impl<'db> ConstId, StaticId, AnonConstId<'db> for GeneralConstId<'db>);

impl<'db> GeneralConstId<'db> {
    pub fn generic_def(self, db: &'db dyn HirDatabase) -> Option<GenericDefId> {
        match self {
            GeneralConstId::ConstId(it) => Some(it.into()),
            GeneralConstId::StaticId(it) => Some(it.into()),
            GeneralConstId::AnonConstId(it) => Some(it.loc(db).owner.generic_def(db)),
        }
    }

    pub fn name(self, db: &'db dyn SourceDatabase) -> String {
        match self {
            GeneralConstId::StaticId(it) => {
                StaticSignature::of(db, it).name.display(db, Edition::CURRENT).to_string()
            }
            GeneralConstId::ConstId(const_id) => {
                ConstSignature::of(db, const_id).name.as_ref().map_or_else(
                    || "_".to_owned(),
                    |name| name.display(db, Edition::CURRENT).to_string(),
                )
            }
            GeneralConstId::AnonConstId(_) => "{const}".to_owned(),
        }
    }
}
