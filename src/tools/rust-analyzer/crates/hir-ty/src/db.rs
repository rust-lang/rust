//! The home of `HirDatabase`, which is the Salsa database containing all the
//! type inference-related queries.

use base_db::{Crate, target::TargetLoadError};
use hir_def::{
    AdtId, BlockId, CallableDefId, ConstParamId, DefWithBodyId, EnumVariantId, FunctionId,
    GeneralConstId, GenericDefId, ImplId, LifetimeParamId, LocalFieldId, StaticId, TraitId,
    TypeAliasId, TypeOrConstParamId, VariantId, db::DefDatabase, hir::ExprId,
    layout::TargetDataLayout,
};
use hir_expand::name::Name;
use la_arena::ArenaMap;
use salsa::plumbing::AsId;
use smallvec::SmallVec;
use triomphe::Arc;

use crate::{
    ImplTraitId, InferenceResult, TraitEnvironment, TyDefId, ValueTyDefId,
    consteval::ConstEvalError,
    dyn_compatibility::DynCompatibilityViolation,
    layout::{Layout, LayoutError},
    lower::{Diagnostics, GenericDefaults, GenericPredicates, ImplTraits},
    method_resolution::{InherentImpls, TraitImpls, TyFingerprint},
    mir::{BorrowckResult, MirBody, MirLowerError},
    next_solver::{Const, EarlyBinder, GenericArgs, PolyFnSig, TraitRef, Ty, VariancesOf},
};

#[query_group::query_group]
pub trait HirDatabase: DefDatabase + std::fmt::Debug {
    #[salsa::invoke(crate::infer::infer_query)]
    #[salsa::cycle(cycle_result = crate::infer::infer_cycle_result)]
    fn infer<'db>(&'db self, def: DefWithBodyId) -> Arc<InferenceResult<'db>>;

    // region:mir

    #[salsa::invoke(crate::mir::mir_body_query)]
    #[salsa::cycle(cycle_result = crate::mir::mir_body_cycle_result)]
    fn mir_body<'db>(
        &'db self,
        def: DefWithBodyId,
    ) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>>;

    #[salsa::invoke(crate::mir::mir_body_for_closure_query)]
    fn mir_body_for_closure<'db>(
        &'db self,
        def: InternedClosureId,
    ) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>>;

    #[salsa::invoke(crate::mir::monomorphized_mir_body_query)]
    #[salsa::cycle(cycle_result = crate::mir::monomorphized_mir_body_cycle_result)]
    fn monomorphized_mir_body<'db>(
        &'db self,
        def: DefWithBodyId,
        subst: GenericArgs<'db>,
        env: Arc<TraitEnvironment<'db>>,
    ) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>>;

    #[salsa::invoke(crate::mir::monomorphized_mir_body_for_closure_query)]
    fn monomorphized_mir_body_for_closure<'db>(
        &'db self,
        def: InternedClosureId,
        subst: GenericArgs<'db>,
        env: Arc<TraitEnvironment<'db>>,
    ) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>>;

    #[salsa::invoke(crate::mir::borrowck_query)]
    #[salsa::lru(2024)]
    fn borrowck<'db>(
        &'db self,
        def: DefWithBodyId,
    ) -> Result<Arc<[BorrowckResult<'db>]>, MirLowerError<'db>>;

    #[salsa::invoke(crate::consteval::const_eval_query)]
    #[salsa::cycle(cycle_result = crate::consteval::const_eval_cycle_result)]
    fn const_eval<'db>(
        &'db self,
        def: GeneralConstId,
        subst: GenericArgs<'db>,
        trait_env: Option<Arc<TraitEnvironment<'db>>>,
    ) -> Result<Const<'db>, ConstEvalError<'db>>;

    #[salsa::invoke(crate::consteval::const_eval_static_query)]
    #[salsa::cycle(cycle_result = crate::consteval::const_eval_static_cycle_result)]
    fn const_eval_static<'db>(&'db self, def: StaticId) -> Result<Const<'db>, ConstEvalError<'db>>;

    #[salsa::invoke(crate::consteval::const_eval_discriminant_variant)]
    #[salsa::cycle(cycle_result = crate::consteval::const_eval_discriminant_cycle_result)]
    fn const_eval_discriminant<'db>(
        &'db self,
        def: EnumVariantId,
    ) -> Result<i128, ConstEvalError<'db>>;

    #[salsa::invoke(crate::method_resolution::lookup_impl_method_query)]
    #[salsa::transparent]
    fn lookup_impl_method<'db>(
        &'db self,
        env: Arc<TraitEnvironment<'db>>,
        func: FunctionId,
        fn_subst: GenericArgs<'db>,
    ) -> (FunctionId, GenericArgs<'db>);

    // endregion:mir

    #[salsa::invoke(crate::layout::layout_of_adt_query)]
    #[salsa::cycle(cycle_result = crate::layout::layout_of_adt_cycle_result)]
    fn layout_of_adt<'db>(
        &'db self,
        def: AdtId,
        args: GenericArgs<'db>,
        trait_env: Arc<TraitEnvironment<'db>>,
    ) -> Result<Arc<Layout>, LayoutError>;

    #[salsa::invoke(crate::layout::layout_of_ty_query)]
    #[salsa::cycle(cycle_result = crate::layout::layout_of_ty_cycle_result)]
    fn layout_of_ty<'db>(
        &'db self,
        ty: Ty<'db>,
        env: Arc<TraitEnvironment<'db>>,
    ) -> Result<Arc<Layout>, LayoutError>;

    #[salsa::invoke(crate::layout::target_data_layout_query)]
    fn target_data_layout(&self, krate: Crate) -> Result<Arc<TargetDataLayout>, TargetLoadError>;

    #[salsa::invoke(crate::dyn_compatibility::dyn_compatibility_of_trait_query)]
    fn dyn_compatibility_of_trait(&self, trait_: TraitId) -> Option<DynCompatibilityViolation>;

    #[salsa::invoke(crate::lower::ty_query)]
    #[salsa::transparent]
    fn ty<'db>(&'db self, def: TyDefId) -> EarlyBinder<'db, Ty<'db>>;

    #[salsa::invoke(crate::lower::type_for_type_alias_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::type_for_type_alias_with_diagnostics_cycle_result)]
    fn type_for_type_alias_with_diagnostics<'db>(
        &'db self,
        def: TypeAliasId,
    ) -> (EarlyBinder<'db, Ty<'db>>, Diagnostics);

    /// Returns the type of the value of the given constant, or `None` if the `ValueTyDefId` is
    /// a `StructId` or `EnumVariantId` with a record constructor.
    #[salsa::invoke(crate::lower::value_ty_query)]
    fn value_ty<'db>(&'db self, def: ValueTyDefId) -> Option<EarlyBinder<'db, Ty<'db>>>;

    #[salsa::invoke(crate::lower::impl_self_ty_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::impl_self_ty_with_diagnostics_cycle_result)]
    fn impl_self_ty_with_diagnostics<'db>(
        &'db self,
        def: ImplId,
    ) -> (EarlyBinder<'db, Ty<'db>>, Diagnostics);

    #[salsa::invoke(crate::lower::impl_self_ty_query)]
    #[salsa::transparent]
    fn impl_self_ty<'db>(&'db self, def: ImplId) -> EarlyBinder<'db, Ty<'db>>;

    // FIXME: Make this a non-interned query.
    #[salsa::invoke_interned(crate::lower::const_param_ty_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::const_param_ty_with_diagnostics_cycle_result)]
    fn const_param_ty_with_diagnostics<'db>(&'db self, def: ConstParamId)
    -> (Ty<'db>, Diagnostics);

    #[salsa::invoke(crate::lower::const_param_ty_query)]
    #[salsa::transparent]
    fn const_param_ty_ns<'db>(&'db self, def: ConstParamId) -> Ty<'db>;

    #[salsa::invoke(crate::lower::impl_trait_with_diagnostics_query)]
    fn impl_trait_with_diagnostics<'db>(
        &'db self,
        def: ImplId,
    ) -> Option<(EarlyBinder<'db, TraitRef<'db>>, Diagnostics)>;

    #[salsa::invoke(crate::lower::impl_trait_query)]
    #[salsa::transparent]
    fn impl_trait<'db>(&'db self, def: ImplId) -> Option<EarlyBinder<'db, TraitRef<'db>>>;

    #[salsa::invoke(crate::lower::field_types_with_diagnostics_query)]
    fn field_types_with_diagnostics<'db>(
        &'db self,
        var: VariantId,
    ) -> (Arc<ArenaMap<LocalFieldId, EarlyBinder<'db, Ty<'db>>>>, Diagnostics);

    #[salsa::invoke(crate::lower::field_types_query)]
    #[salsa::transparent]
    fn field_types<'db>(
        &'db self,
        var: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, EarlyBinder<'db, Ty<'db>>>>;

    #[salsa::invoke(crate::lower::callable_item_signature_query)]
    fn callable_item_signature<'db>(
        &'db self,
        def: CallableDefId,
    ) -> EarlyBinder<'db, PolyFnSig<'db>>;

    #[salsa::invoke(crate::lower::return_type_impl_traits)]
    fn return_type_impl_traits<'db>(
        &'db self,
        def: FunctionId,
    ) -> Option<Arc<EarlyBinder<'db, ImplTraits<'db>>>>;

    #[salsa::invoke(crate::lower::type_alias_impl_traits)]
    fn type_alias_impl_traits<'db>(
        &'db self,
        def: TypeAliasId,
    ) -> Option<Arc<EarlyBinder<'db, ImplTraits<'db>>>>;

    #[salsa::invoke(crate::lower::generic_predicates_without_parent_with_diagnostics_query)]
    fn generic_predicates_without_parent_with_diagnostics<'db>(
        &'db self,
        def: GenericDefId,
    ) -> (GenericPredicates<'db>, Diagnostics);

    #[salsa::invoke(crate::lower::generic_predicates_without_parent_query)]
    #[salsa::transparent]
    fn generic_predicates_without_parent<'db>(
        &'db self,
        def: GenericDefId,
    ) -> GenericPredicates<'db>;

    #[salsa::invoke(crate::lower::generic_predicates_for_param_query)]
    #[salsa::cycle(cycle_result = crate::lower::generic_predicates_for_param_cycle_result)]
    fn generic_predicates_for_param<'db>(
        &'db self,
        def: GenericDefId,
        param_id: TypeOrConstParamId,
        assoc_name: Option<Name>,
    ) -> GenericPredicates<'db>;

    #[salsa::invoke(crate::lower::generic_predicates_query)]
    fn generic_predicates<'db>(&'db self, def: GenericDefId) -> GenericPredicates<'db>;

    #[salsa::invoke(crate::lower::trait_environment_for_body_query)]
    #[salsa::transparent]
    fn trait_environment_for_body<'db>(&'db self, def: DefWithBodyId)
    -> Arc<TraitEnvironment<'db>>;

    #[salsa::invoke(crate::lower::trait_environment_query)]
    fn trait_environment<'db>(&'db self, def: GenericDefId) -> Arc<TraitEnvironment<'db>>;

    #[salsa::invoke(crate::lower::generic_defaults_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::generic_defaults_with_diagnostics_cycle_result)]
    fn generic_defaults_with_diagnostics<'db>(
        &'db self,
        def: GenericDefId,
    ) -> (GenericDefaults<'db>, Diagnostics);

    /// This returns an empty list if no parameter has default.
    ///
    /// The binders of the returned defaults are only up to (not including) this parameter.
    #[salsa::invoke(crate::lower::generic_defaults_query)]
    #[salsa::transparent]
    fn generic_defaults<'db>(&'db self, def: GenericDefId) -> GenericDefaults<'db>;

    #[salsa::invoke(InherentImpls::inherent_impls_in_crate_query)]
    fn inherent_impls_in_crate(&self, krate: Crate) -> Arc<InherentImpls>;

    #[salsa::invoke(InherentImpls::inherent_impls_in_block_query)]
    fn inherent_impls_in_block(&self, block: BlockId) -> Option<Arc<InherentImpls>>;

    /// Collects all crates in the dependency graph that have impls for the
    /// given fingerprint. This is only used for primitive types and types
    /// annotated with `rustc_has_incoherent_inherent_impls`; for other types
    /// we just look at the crate where the type is defined.
    #[salsa::invoke(crate::method_resolution::incoherent_inherent_impl_crates)]
    fn incoherent_inherent_impl_crates(
        &self,
        krate: Crate,
        fp: TyFingerprint,
    ) -> SmallVec<[Crate; 2]>;

    #[salsa::invoke(TraitImpls::trait_impls_in_crate_query)]
    fn trait_impls_in_crate(&self, krate: Crate) -> Arc<TraitImpls>;

    #[salsa::invoke(TraitImpls::trait_impls_in_block_query)]
    fn trait_impls_in_block(&self, block: BlockId) -> Option<Arc<TraitImpls>>;

    #[salsa::invoke(TraitImpls::trait_impls_in_deps_query)]
    fn trait_impls_in_deps(&self, krate: Crate) -> Arc<[Arc<TraitImpls>]>;

    // Interned IDs for solver integration
    #[salsa::interned]
    fn intern_impl_trait_id(&self, id: ImplTraitId<'_>) -> InternedOpaqueTyId;

    #[salsa::interned]
    fn intern_closure(&self, id: InternedClosure) -> InternedClosureId;

    #[salsa::interned]
    fn intern_coroutine(&self, id: InternedCoroutine) -> InternedCoroutineId;

    #[salsa::invoke(crate::variance::variances_of)]
    #[salsa::cycle(
        // cycle_fn = crate::variance::variances_of_cycle_fn,
        // cycle_initial = crate::variance::variances_of_cycle_initial,
        cycle_result = crate::variance::variances_of_cycle_initial,
    )]
    fn variances_of<'db>(&'db self, def: GenericDefId) -> VariancesOf<'db>;
}

#[test]
fn hir_database_is_dyn_compatible() {
    fn _assert_dyn_compatible(_: &dyn HirDatabase) {}
}

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedTypeOrConstParamId {
    /// This stores the param and its index.
    pub loc: (TypeOrConstParamId, u32),
}

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedLifetimeParamId {
    /// This stores the param and its index.
    pub loc: (LifetimeParamId, u32),
}

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedConstParamId {
    pub loc: ConstParamId,
}

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedOpaqueTyId {
    pub loc: ImplTraitId<'db>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternedClosure(pub DefWithBodyId, pub ExprId);

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedClosureId {
    pub loc: InternedClosure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternedCoroutine(pub DefWithBodyId, pub ExprId);

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedCoroutineId {
    pub loc: InternedCoroutine,
}
