//! The home of `HirDatabase`, which is the Salsa database containing all the
//! type inference-related queries.

use std::sync;

use base_db::{Crate, impl_intern_key};
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
    Binders, Const, ImplTraitId, ImplTraits, InferenceResult, Interner, PolyFnSig, Substitution,
    TraitEnvironment, TraitRef, Ty, TyDefId, ValueTyDefId, chalk_db,
    consteval::ConstEvalError,
    drop::DropGlue,
    dyn_compatibility::DynCompatibilityViolation,
    layout::{Layout, LayoutError},
    lower::{Diagnostics, GenericDefaults, GenericPredicates},
    method_resolution::{InherentImpls, TraitImpls, TyFingerprint},
    mir::{BorrowckResult, MirBody, MirLowerError},
};

#[query_group::query_group]
pub trait HirDatabase: DefDatabase + std::fmt::Debug {
    #[salsa::invoke(crate::infer::infer_query)]
    #[salsa::cycle(cycle_result = crate::infer::infer_cycle_result)]
    fn infer(&self, def: DefWithBodyId) -> Arc<InferenceResult>;

    // region:mir

    #[salsa::invoke(crate::mir::mir_body_query)]
    #[salsa::cycle(cycle_result = crate::mir::mir_body_cycle_result)]
    fn mir_body(&self, def: DefWithBodyId) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::mir_body_for_closure_query)]
    fn mir_body_for_closure(&self, def: InternedClosureId) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::monomorphized_mir_body_query)]
    #[salsa::cycle(cycle_result = crate::mir::monomorphized_mir_body_cycle_result)]
    fn monomorphized_mir_body(
        &self,
        def: DefWithBodyId,
        subst: Substitution,
        env: Arc<TraitEnvironment>,
    ) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::monomorphized_mir_body_for_closure_query)]
    fn monomorphized_mir_body_for_closure(
        &self,
        def: InternedClosureId,
        subst: Substitution,
        env: Arc<TraitEnvironment>,
    ) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::borrowck_query)]
    #[salsa::lru(2024)]
    fn borrowck(&self, def: DefWithBodyId) -> Result<Arc<[BorrowckResult]>, MirLowerError>;

    #[salsa::invoke(crate::consteval::const_eval_query)]
    #[salsa::cycle(cycle_result = crate::consteval::const_eval_cycle_result)]
    fn const_eval(
        &self,
        def: GeneralConstId,
        subst: Substitution,
        trait_env: Option<Arc<TraitEnvironment>>,
    ) -> Result<Const, ConstEvalError>;

    #[salsa::invoke(crate::consteval::const_eval_static_query)]
    #[salsa::cycle(cycle_result = crate::consteval::const_eval_static_cycle_result)]
    fn const_eval_static(&self, def: StaticId) -> Result<Const, ConstEvalError>;

    #[salsa::invoke(crate::consteval::const_eval_discriminant_variant)]
    #[salsa::cycle(cycle_result = crate::consteval::const_eval_discriminant_cycle_result)]
    fn const_eval_discriminant(&self, def: EnumVariantId) -> Result<i128, ConstEvalError>;

    #[salsa::invoke(crate::method_resolution::lookup_impl_method_query)]
    fn lookup_impl_method(
        &self,
        env: Arc<TraitEnvironment>,
        func: FunctionId,
        fn_subst: Substitution,
    ) -> (FunctionId, Substitution);

    // endregion:mir

    #[salsa::invoke(crate::layout::layout_of_adt_query)]
    #[salsa::cycle(cycle_result = crate::layout::layout_of_adt_cycle_result)]
    fn layout_of_adt(
        &self,
        def: AdtId,
        subst: Substitution,
        env: Arc<TraitEnvironment>,
    ) -> Result<Arc<Layout>, LayoutError>;

    #[salsa::invoke(crate::layout::layout_of_ty_query)]
    #[salsa::cycle(cycle_result = crate::layout::layout_of_ty_cycle_result)]
    fn layout_of_ty(&self, ty: Ty, env: Arc<TraitEnvironment>) -> Result<Arc<Layout>, LayoutError>;

    #[salsa::invoke(crate::layout::target_data_layout_query)]
    fn target_data_layout(&self, krate: Crate) -> Result<Arc<TargetDataLayout>, Arc<str>>;

    #[salsa::invoke(crate::dyn_compatibility::dyn_compatibility_of_trait_query)]
    fn dyn_compatibility_of_trait(&self, trait_: TraitId) -> Option<DynCompatibilityViolation>;

    #[salsa::invoke(crate::lower::ty_query)]
    #[salsa::transparent]
    fn ty(&self, def: TyDefId) -> Binders<Ty>;

    #[salsa::invoke(crate::lower::type_for_type_alias_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::type_for_type_alias_with_diagnostics_cycle_result)]
    fn type_for_type_alias_with_diagnostics(&self, def: TypeAliasId) -> (Binders<Ty>, Diagnostics);

    /// Returns the type of the value of the given constant, or `None` if the `ValueTyDefId` is
    /// a `StructId` or `EnumVariantId` with a record constructor.
    #[salsa::invoke(crate::lower::value_ty_query)]
    fn value_ty(&self, def: ValueTyDefId) -> Option<Binders<Ty>>;

    #[salsa::invoke(crate::lower::impl_self_ty_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::impl_self_ty_with_diagnostics_cycle_result)]
    fn impl_self_ty_with_diagnostics(&self, def: ImplId) -> (Binders<Ty>, Diagnostics);

    #[salsa::invoke(crate::lower::impl_self_ty_query)]
    #[salsa::transparent]
    fn impl_self_ty(&self, def: ImplId) -> Binders<Ty>;

    // FIXME: Make this a non-interned query.
    #[salsa::invoke_interned(crate::lower::const_param_ty_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::const_param_ty_with_diagnostics_cycle_result)]
    fn const_param_ty_with_diagnostics(&self, def: ConstParamId) -> (Ty, Diagnostics);

    #[salsa::invoke(crate::lower::const_param_ty_query)]
    #[salsa::transparent]
    fn const_param_ty(&self, def: ConstParamId) -> Ty;

    #[salsa::invoke(crate::lower::impl_trait_with_diagnostics_query)]
    fn impl_trait_with_diagnostics(&self, def: ImplId) -> Option<(Binders<TraitRef>, Diagnostics)>;

    #[salsa::invoke(crate::lower::impl_trait_query)]
    #[salsa::transparent]
    fn impl_trait(&self, def: ImplId) -> Option<Binders<TraitRef>>;

    #[salsa::invoke(crate::lower::field_types_with_diagnostics_query)]
    fn field_types_with_diagnostics(
        &self,
        var: VariantId,
    ) -> (Arc<ArenaMap<LocalFieldId, Binders<Ty>>>, Diagnostics);

    #[salsa::invoke(crate::lower::field_types_query)]
    #[salsa::transparent]
    fn field_types(&self, var: VariantId) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>>;

    #[salsa::invoke(crate::lower::callable_item_signature_query)]
    fn callable_item_signature(&self, def: CallableDefId) -> PolyFnSig;

    #[salsa::invoke(crate::lower::return_type_impl_traits)]
    fn return_type_impl_traits(&self, def: FunctionId) -> Option<Arc<Binders<ImplTraits>>>;

    #[salsa::invoke(crate::lower::type_alias_impl_traits)]
    fn type_alias_impl_traits(&self, def: TypeAliasId) -> Option<Arc<Binders<ImplTraits>>>;

    #[salsa::invoke(crate::lower::generic_predicates_for_param_query)]
    #[salsa::cycle(cycle_result = crate::lower::generic_predicates_for_param_cycle_result)]
    fn generic_predicates_for_param(
        &self,
        def: GenericDefId,
        param_id: TypeOrConstParamId,
        assoc_name: Option<Name>,
    ) -> GenericPredicates;

    #[salsa::invoke(crate::lower::generic_predicates_query)]
    fn generic_predicates(&self, def: GenericDefId) -> GenericPredicates;

    #[salsa::invoke(crate::lower::generic_predicates_without_parent_with_diagnostics_query)]
    fn generic_predicates_without_parent_with_diagnostics(
        &self,
        def: GenericDefId,
    ) -> (GenericPredicates, Diagnostics);

    #[salsa::invoke(crate::lower::generic_predicates_without_parent_query)]
    #[salsa::transparent]
    fn generic_predicates_without_parent(&self, def: GenericDefId) -> GenericPredicates;

    #[salsa::invoke(crate::lower::trait_environment_for_body_query)]
    #[salsa::transparent]
    fn trait_environment_for_body(&self, def: DefWithBodyId) -> Arc<TraitEnvironment>;

    #[salsa::invoke(crate::lower::trait_environment_query)]
    fn trait_environment(&self, def: GenericDefId) -> Arc<TraitEnvironment>;

    #[salsa::invoke(crate::lower::generic_defaults_with_diagnostics_query)]
    #[salsa::cycle(cycle_result = crate::lower::generic_defaults_with_diagnostics_cycle_result)]
    fn generic_defaults_with_diagnostics(
        &self,
        def: GenericDefId,
    ) -> (GenericDefaults, Diagnostics);

    /// This returns an empty list if no parameter has default.
    ///
    /// The binders of the returned defaults are only up to (not including) this parameter.
    #[salsa::invoke(crate::lower::generic_defaults_query)]
    #[salsa::transparent]
    fn generic_defaults(&self, def: GenericDefId) -> GenericDefaults;

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

    // Interned IDs for Chalk integration
    #[salsa::interned]
    fn intern_impl_trait_id(&self, id: ImplTraitId) -> InternedOpaqueTyId;

    #[salsa::interned]
    fn intern_closure(&self, id: InternedClosure) -> InternedClosureId;

    #[salsa::interned]
    fn intern_coroutine(&self, id: InternedCoroutine) -> InternedCoroutineId;

    #[salsa::invoke(chalk_db::associated_ty_data_query)]
    fn associated_ty_data(&self, id: TypeAliasId) -> sync::Arc<chalk_db::AssociatedTyDatum>;

    #[salsa::invoke(chalk_db::trait_datum_query)]
    fn trait_datum(
        &self,
        krate: Crate,
        trait_id: chalk_db::TraitId,
    ) -> sync::Arc<chalk_db::TraitDatum>;

    #[salsa::invoke(chalk_db::adt_datum_query)]
    fn adt_datum(&self, krate: Crate, struct_id: chalk_db::AdtId) -> sync::Arc<chalk_db::AdtDatum>;

    #[salsa::invoke(chalk_db::impl_datum_query)]
    fn impl_datum(&self, krate: Crate, impl_id: chalk_db::ImplId)
    -> sync::Arc<chalk_db::ImplDatum>;

    #[salsa::invoke(chalk_db::fn_def_datum_query)]
    fn fn_def_datum(&self, fn_def_id: CallableDefId) -> sync::Arc<chalk_db::FnDefDatum>;

    #[salsa::invoke(chalk_db::fn_def_variance_query)]
    fn fn_def_variance(&self, fn_def_id: CallableDefId) -> chalk_db::Variances;

    #[salsa::invoke(chalk_db::adt_variance_query)]
    fn adt_variance(&self, adt_id: AdtId) -> chalk_db::Variances;

    #[salsa::invoke(crate::variance::variances_of)]
    #[salsa::cycle(
        // cycle_fn = crate::variance::variances_of_cycle_fn,
        // cycle_initial = crate::variance::variances_of_cycle_initial,
        cycle_result = crate::variance::variances_of_cycle_initial,
    )]
    fn variances_of(&self, def: GenericDefId) -> Option<Arc<[crate::variance::Variance]>>;

    #[salsa::invoke(chalk_db::associated_ty_value_query)]
    fn associated_ty_value(
        &self,
        krate: Crate,
        id: chalk_db::AssociatedTyValueId,
    ) -> sync::Arc<chalk_db::AssociatedTyValue>;

    #[salsa::invoke(crate::traits::normalize_projection_query)]
    #[salsa::transparent]
    fn normalize_projection(
        &self,
        projection: crate::ProjectionTy,
        env: Arc<TraitEnvironment>,
    ) -> Ty;

    #[salsa::invoke(crate::traits::trait_solve_query)]
    fn trait_solve(
        &self,
        krate: Crate,
        block: Option<BlockId>,
        goal: crate::Canonical<crate::InEnvironment<crate::Goal>>,
    ) -> Option<crate::Solution>;

    #[salsa::invoke(chalk_db::program_clauses_for_chalk_env_query)]
    fn program_clauses_for_chalk_env(
        &self,
        krate: Crate,
        block: Option<BlockId>,
        env: chalk_ir::Environment<Interner>,
    ) -> chalk_ir::ProgramClauses<Interner>;

    #[salsa::invoke(crate::drop::has_drop_glue)]
    #[salsa::cycle(cycle_result = crate::drop::has_drop_glue_cycle_result)]
    fn has_drop_glue(&self, ty: Ty, env: Arc<TraitEnvironment>) -> DropGlue;
}

#[test]
fn hir_database_is_dyn_compatible() {
    fn _assert_dyn_compatible(_: &dyn HirDatabase) {}
}

#[salsa_macros::interned(no_lifetime, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedTypeOrConstParamId {
    pub loc: TypeOrConstParamId,
}
impl ::std::fmt::Debug for InternedTypeOrConstParamId {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        f.debug_tuple(stringify!(InternedTypeOrConstParamId))
            .field(&format_args!("{:04x}", self.0.index()))
            .finish()
    }
}

#[salsa_macros::interned(no_lifetime, revisions = usize::MAX)]
#[derive(PartialOrd, Ord)]
pub struct InternedLifetimeParamId {
    pub loc: LifetimeParamId,
}
impl ::std::fmt::Debug for InternedLifetimeParamId {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        f.debug_tuple(stringify!(InternedLifetimeParamId))
            .field(&format_args!("{:04x}", self.0.index()))
            .finish()
    }
}

impl_intern_key!(InternedConstParamId, ConstParamId);

impl_intern_key!(InternedOpaqueTyId, ImplTraitId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternedClosure(pub DefWithBodyId, pub ExprId);
impl_intern_key!(InternedClosureId, InternedClosure);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternedCoroutine(pub DefWithBodyId, pub ExprId);
impl_intern_key!(InternedCoroutineId, InternedCoroutine);
