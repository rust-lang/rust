//! The home of `HirDatabase`, which is the Salsa database containing all the
//! type inference-related queries.

use std::sync;

use base_db::{impl_intern_key, salsa, CrateId, Upcast};
use hir_def::{
    db::DefDatabase, hir::ExprId, layout::TargetDataLayout, AdtId, BlockId, ConstParamId,
    DefWithBodyId, EnumVariantId, FunctionId, GeneralConstId, GenericDefId, ImplId,
    LifetimeParamId, LocalFieldId, StaticId, TypeOrConstParamId, VariantId,
};
use la_arena::ArenaMap;
use smallvec::SmallVec;
use triomphe::Arc;

use crate::{
    chalk_db,
    consteval::ConstEvalError,
    layout::{Layout, LayoutError},
    method_resolution::{InherentImpls, TraitImpls, TyFingerprint},
    mir::{BorrowckResult, MirBody, MirLowerError},
    Binders, CallableDefId, ClosureId, Const, FnDefId, GenericArg, ImplTraitId, InferenceResult,
    Interner, PolyFnSig, QuantifiedWhereClause, ReturnTypeImplTraits, Substitution, TraitRef, Ty,
    TyDefId, ValueTyDefId,
};
use hir_expand::name::Name;

#[salsa::query_group(HirDatabaseStorage)]
pub trait HirDatabase: DefDatabase + Upcast<dyn DefDatabase> {
    #[salsa::invoke(infer_wait)]
    #[salsa::transparent]
    fn infer(&self, def: DefWithBodyId) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::infer::infer_query)]
    fn infer_query(&self, def: DefWithBodyId) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::mir::mir_body_query)]
    #[salsa::cycle(crate::mir::mir_body_recover)]
    fn mir_body(&self, def: DefWithBodyId) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::mir_body_for_closure_query)]
    fn mir_body_for_closure(&self, def: ClosureId) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::monomorphized_mir_body_query)]
    #[salsa::cycle(crate::mir::monomorphized_mir_body_recover)]
    fn monomorphized_mir_body(
        &self,
        def: DefWithBodyId,
        subst: Substitution,
        env: Arc<crate::TraitEnvironment>,
    ) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::monomorphized_mir_body_for_closure_query)]
    fn monomorphized_mir_body_for_closure(
        &self,
        def: ClosureId,
        subst: Substitution,
        env: Arc<crate::TraitEnvironment>,
    ) -> Result<Arc<MirBody>, MirLowerError>;

    #[salsa::invoke(crate::mir::borrowck_query)]
    fn borrowck(&self, def: DefWithBodyId) -> Result<Arc<[BorrowckResult]>, MirLowerError>;

    #[salsa::invoke(crate::lower::ty_query)]
    #[salsa::cycle(crate::lower::ty_recover)]
    fn ty(&self, def: TyDefId) -> Binders<Ty>;

    #[salsa::invoke(crate::lower::value_ty_query)]
    fn value_ty(&self, def: ValueTyDefId) -> Binders<Ty>;

    #[salsa::invoke(crate::lower::impl_self_ty_query)]
    #[salsa::cycle(crate::lower::impl_self_ty_recover)]
    fn impl_self_ty(&self, def: ImplId) -> Binders<Ty>;

    #[salsa::invoke(crate::lower::const_param_ty_query)]
    fn const_param_ty(&self, def: ConstParamId) -> Ty;

    #[salsa::invoke(crate::consteval::const_eval_query)]
    #[salsa::cycle(crate::consteval::const_eval_recover)]
    fn const_eval(&self, def: GeneralConstId, subst: Substitution)
        -> Result<Const, ConstEvalError>;

    #[salsa::invoke(crate::consteval::const_eval_static_query)]
    #[salsa::cycle(crate::consteval::const_eval_static_recover)]
    fn const_eval_static(&self, def: StaticId) -> Result<Const, ConstEvalError>;

    #[salsa::invoke(crate::consteval::const_eval_discriminant_variant)]
    #[salsa::cycle(crate::consteval::const_eval_discriminant_recover)]
    fn const_eval_discriminant(&self, def: EnumVariantId) -> Result<i128, ConstEvalError>;

    #[salsa::invoke(crate::lower::impl_trait_query)]
    fn impl_trait(&self, def: ImplId) -> Option<Binders<TraitRef>>;

    #[salsa::invoke(crate::lower::field_types_query)]
    fn field_types(&self, var: VariantId) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>>;

    #[salsa::invoke(crate::layout::layout_of_adt_query)]
    #[salsa::cycle(crate::layout::layout_of_adt_recover)]
    fn layout_of_adt(
        &self,
        def: AdtId,
        subst: Substitution,
        krate: CrateId,
    ) -> Result<Arc<Layout>, LayoutError>;

    #[salsa::invoke(crate::layout::layout_of_ty_query)]
    #[salsa::cycle(crate::layout::layout_of_ty_recover)]
    fn layout_of_ty(&self, ty: Ty, krate: CrateId) -> Result<Arc<Layout>, LayoutError>;

    #[salsa::invoke(crate::layout::target_data_layout_query)]
    fn target_data_layout(&self, krate: CrateId) -> Option<Arc<TargetDataLayout>>;

    #[salsa::invoke(crate::method_resolution::lookup_impl_method_query)]
    fn lookup_impl_method(
        &self,
        env: Arc<crate::TraitEnvironment>,
        func: FunctionId,
        fn_subst: Substitution,
    ) -> (FunctionId, Substitution);

    #[salsa::invoke(crate::lower::callable_item_sig)]
    fn callable_item_signature(&self, def: CallableDefId) -> PolyFnSig;

    #[salsa::invoke(crate::lower::return_type_impl_traits)]
    fn return_type_impl_traits(
        &self,
        def: FunctionId,
    ) -> Option<Arc<Binders<ReturnTypeImplTraits>>>;

    #[salsa::invoke(crate::lower::generic_predicates_for_param_query)]
    #[salsa::cycle(crate::lower::generic_predicates_for_param_recover)]
    fn generic_predicates_for_param(
        &self,
        def: GenericDefId,
        param_id: TypeOrConstParamId,
        assoc_name: Option<Name>,
    ) -> Arc<[Binders<QuantifiedWhereClause>]>;

    #[salsa::invoke(crate::lower::generic_predicates_query)]
    fn generic_predicates(&self, def: GenericDefId) -> Arc<[Binders<QuantifiedWhereClause>]>;

    #[salsa::invoke(crate::lower::trait_environment_for_body_query)]
    #[salsa::transparent]
    fn trait_environment_for_body(&self, def: DefWithBodyId) -> Arc<crate::TraitEnvironment>;

    #[salsa::invoke(crate::lower::trait_environment_query)]
    fn trait_environment(&self, def: GenericDefId) -> Arc<crate::TraitEnvironment>;

    #[salsa::invoke(crate::lower::generic_defaults_query)]
    #[salsa::cycle(crate::lower::generic_defaults_recover)]
    fn generic_defaults(&self, def: GenericDefId) -> Arc<[Binders<GenericArg>]>;

    #[salsa::invoke(InherentImpls::inherent_impls_in_crate_query)]
    fn inherent_impls_in_crate(&self, krate: CrateId) -> Arc<InherentImpls>;

    #[salsa::invoke(InherentImpls::inherent_impls_in_block_query)]
    fn inherent_impls_in_block(&self, block: BlockId) -> Arc<InherentImpls>;

    /// Collects all crates in the dependency graph that have impls for the
    /// given fingerprint. This is only used for primitive types and types
    /// annotated with `rustc_has_incoherent_inherent_impls`; for other types
    /// we just look at the crate where the type is defined.
    #[salsa::invoke(crate::method_resolution::incoherent_inherent_impl_crates)]
    fn incoherent_inherent_impl_crates(
        &self,
        krate: CrateId,
        fp: TyFingerprint,
    ) -> SmallVec<[CrateId; 2]>;

    #[salsa::invoke(TraitImpls::trait_impls_in_crate_query)]
    fn trait_impls_in_crate(&self, krate: CrateId) -> Arc<TraitImpls>;

    #[salsa::invoke(TraitImpls::trait_impls_in_block_query)]
    fn trait_impls_in_block(&self, block: BlockId) -> Arc<TraitImpls>;

    #[salsa::invoke(TraitImpls::trait_impls_in_deps_query)]
    fn trait_impls_in_deps(&self, krate: CrateId) -> Arc<[Arc<TraitImpls>]>;

    // Interned IDs for Chalk integration
    #[salsa::interned]
    fn intern_callable_def(&self, callable_def: CallableDefId) -> InternedCallableDefId;
    #[salsa::interned]
    fn intern_type_or_const_param_id(
        &self,
        param_id: TypeOrConstParamId,
    ) -> InternedTypeOrConstParamId;
    #[salsa::interned]
    fn intern_lifetime_param_id(&self, param_id: LifetimeParamId) -> InternedLifetimeParamId;
    #[salsa::interned]
    fn intern_impl_trait_id(&self, id: ImplTraitId) -> InternedOpaqueTyId;
    #[salsa::interned]
    fn intern_closure(&self, id: (DefWithBodyId, ExprId)) -> InternedClosureId;
    #[salsa::interned]
    fn intern_generator(&self, id: (DefWithBodyId, ExprId)) -> InternedGeneratorId;

    #[salsa::invoke(chalk_db::associated_ty_data_query)]
    fn associated_ty_data(
        &self,
        id: chalk_db::AssocTypeId,
    ) -> sync::Arc<chalk_db::AssociatedTyDatum>;

    #[salsa::invoke(chalk_db::trait_datum_query)]
    fn trait_datum(
        &self,
        krate: CrateId,
        trait_id: chalk_db::TraitId,
    ) -> sync::Arc<chalk_db::TraitDatum>;

    #[salsa::invoke(chalk_db::struct_datum_query)]
    fn struct_datum(
        &self,
        krate: CrateId,
        struct_id: chalk_db::AdtId,
    ) -> sync::Arc<chalk_db::StructDatum>;

    #[salsa::invoke(chalk_db::impl_datum_query)]
    fn impl_datum(
        &self,
        krate: CrateId,
        impl_id: chalk_db::ImplId,
    ) -> sync::Arc<chalk_db::ImplDatum>;

    #[salsa::invoke(chalk_db::fn_def_datum_query)]
    fn fn_def_datum(&self, krate: CrateId, fn_def_id: FnDefId) -> sync::Arc<chalk_db::FnDefDatum>;

    #[salsa::invoke(chalk_db::fn_def_variance_query)]
    fn fn_def_variance(&self, fn_def_id: FnDefId) -> chalk_db::Variances;

    #[salsa::invoke(chalk_db::adt_variance_query)]
    fn adt_variance(&self, adt_id: chalk_db::AdtId) -> chalk_db::Variances;

    #[salsa::invoke(chalk_db::associated_ty_value_query)]
    fn associated_ty_value(
        &self,
        krate: CrateId,
        id: chalk_db::AssociatedTyValueId,
    ) -> sync::Arc<chalk_db::AssociatedTyValue>;

    #[salsa::invoke(crate::traits::normalize_projection_query)]
    #[salsa::transparent]
    fn normalize_projection(
        &self,
        projection: crate::ProjectionTy,
        env: Arc<crate::TraitEnvironment>,
    ) -> Ty;

    #[salsa::invoke(trait_solve_wait)]
    #[salsa::transparent]
    fn trait_solve(
        &self,
        krate: CrateId,
        block: Option<BlockId>,
        goal: crate::Canonical<crate::InEnvironment<crate::Goal>>,
    ) -> Option<crate::Solution>;

    #[salsa::invoke(crate::traits::trait_solve_query)]
    fn trait_solve_query(
        &self,
        krate: CrateId,
        block: Option<BlockId>,
        goal: crate::Canonical<crate::InEnvironment<crate::Goal>>,
    ) -> Option<crate::Solution>;

    #[salsa::invoke(chalk_db::program_clauses_for_chalk_env_query)]
    fn program_clauses_for_chalk_env(
        &self,
        krate: CrateId,
        block: Option<BlockId>,
        env: chalk_ir::Environment<Interner>,
    ) -> chalk_ir::ProgramClauses<Interner>;
}

fn infer_wait(db: &dyn HirDatabase, def: DefWithBodyId) -> Arc<InferenceResult> {
    let _p = profile::span("infer:wait").detail(|| match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).name.display(db.upcast()).to_string(),
        DefWithBodyId::StaticId(it) => {
            db.static_data(it).name.clone().display(db.upcast()).to_string()
        }
        DefWithBodyId::ConstId(it) => db
            .const_data(it)
            .name
            .clone()
            .unwrap_or_else(Name::missing)
            .display(db.upcast())
            .to_string(),
        DefWithBodyId::VariantId(it) => {
            db.enum_data(it.parent).variants[it.local_id].name.display(db.upcast()).to_string()
        }
        DefWithBodyId::InTypeConstId(it) => format!("in type const {it:?}"),
    });
    db.infer_query(def)
}

fn trait_solve_wait(
    db: &dyn HirDatabase,
    krate: CrateId,
    block: Option<BlockId>,
    goal: crate::Canonical<crate::InEnvironment<crate::Goal>>,
) -> Option<crate::Solution> {
    let _p = profile::span("trait_solve::wait");
    db.trait_solve_query(krate, block, goal)
}

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedTypeOrConstParamId(salsa::InternId);
impl_intern_key!(InternedTypeOrConstParamId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedLifetimeParamId(salsa::InternId);
impl_intern_key!(InternedLifetimeParamId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedConstParamId(salsa::InternId);
impl_intern_key!(InternedConstParamId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedOpaqueTyId(salsa::InternId);
impl_intern_key!(InternedOpaqueTyId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedClosureId(salsa::InternId);
impl_intern_key!(InternedClosureId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedGeneratorId(salsa::InternId);
impl_intern_key!(InternedGeneratorId);

/// This exists just for Chalk, because Chalk just has a single `FnDefId` where
/// we have different IDs for struct and enum variant constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct InternedCallableDefId(salsa::InternId);
impl_intern_key!(InternedCallableDefId);
