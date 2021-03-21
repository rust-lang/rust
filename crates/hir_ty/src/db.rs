//! FIXME: write short doc here

use std::sync::Arc;

use base_db::{impl_intern_key, salsa, CrateId, Upcast};
use hir_def::{
    db::DefDatabase, expr::ExprId, ConstParamId, DefWithBodyId, FunctionId, GenericDefId, ImplId,
    LocalFieldId, TypeParamId, VariantId,
};
use la_arena::ArenaMap;

use crate::{
    method_resolution::{InherentImpls, TraitImpls},
    traits::chalk,
    Binders, CallableDefId, FnDefId, ImplTraitId, InferenceResult, PolyFnSig,
    QuantifiedWhereClause, ReturnTypeImplTraits, TraitRef, Ty, TyDefId, ValueTyDefId,
};
use hir_expand::name::Name;

#[salsa::query_group(HirDatabaseStorage)]
pub trait HirDatabase: DefDatabase + Upcast<dyn DefDatabase> {
    #[salsa::invoke(infer_wait)]
    #[salsa::transparent]
    fn infer(&self, def: DefWithBodyId) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::infer::infer_query)]
    fn infer_query(&self, def: DefWithBodyId) -> Arc<InferenceResult>;

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

    #[salsa::invoke(crate::lower::impl_trait_query)]
    fn impl_trait(&self, def: ImplId) -> Option<Binders<TraitRef>>;

    #[salsa::invoke(crate::lower::field_types_query)]
    fn field_types(&self, var: VariantId) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>>;

    #[salsa::invoke(crate::callable_item_sig)]
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
        param_id: TypeParamId,
    ) -> Arc<[Binders<QuantifiedWhereClause>]>;

    #[salsa::invoke(crate::lower::generic_predicates_query)]
    fn generic_predicates(&self, def: GenericDefId) -> Arc<[Binders<QuantifiedWhereClause>]>;

    #[salsa::invoke(crate::lower::trait_environment_query)]
    fn trait_environment(&self, def: GenericDefId) -> Arc<crate::TraitEnvironment>;

    #[salsa::invoke(crate::lower::generic_defaults_query)]
    fn generic_defaults(&self, def: GenericDefId) -> Arc<[Binders<Ty>]>;

    #[salsa::invoke(InherentImpls::inherent_impls_in_crate_query)]
    fn inherent_impls_in_crate(&self, krate: CrateId) -> Arc<InherentImpls>;

    #[salsa::invoke(TraitImpls::trait_impls_in_crate_query)]
    fn trait_impls_in_crate(&self, krate: CrateId) -> Arc<TraitImpls>;

    #[salsa::invoke(TraitImpls::trait_impls_in_deps_query)]
    fn trait_impls_in_deps(&self, krate: CrateId) -> Arc<TraitImpls>;

    // Interned IDs for Chalk integration
    #[salsa::interned]
    fn intern_callable_def(&self, callable_def: CallableDefId) -> InternedCallableDefId;
    #[salsa::interned]
    fn intern_type_param_id(&self, param_id: TypeParamId) -> InternedTypeParamId;
    #[salsa::interned]
    fn intern_impl_trait_id(&self, id: ImplTraitId) -> InternedOpaqueTyId;
    #[salsa::interned]
    fn intern_closure(&self, id: (DefWithBodyId, ExprId)) -> InternedClosureId;

    #[salsa::invoke(chalk::associated_ty_data_query)]
    fn associated_ty_data(&self, id: chalk::AssocTypeId) -> Arc<chalk::AssociatedTyDatum>;

    #[salsa::invoke(chalk::trait_datum_query)]
    fn trait_datum(&self, krate: CrateId, trait_id: chalk::TraitId) -> Arc<chalk::TraitDatum>;

    #[salsa::invoke(chalk::struct_datum_query)]
    fn struct_datum(&self, krate: CrateId, struct_id: chalk::AdtId) -> Arc<chalk::StructDatum>;

    #[salsa::invoke(crate::traits::chalk::impl_datum_query)]
    fn impl_datum(&self, krate: CrateId, impl_id: chalk::ImplId) -> Arc<chalk::ImplDatum>;

    #[salsa::invoke(crate::traits::chalk::fn_def_datum_query)]
    fn fn_def_datum(&self, krate: CrateId, fn_def_id: FnDefId) -> Arc<chalk::FnDefDatum>;

    #[salsa::invoke(crate::traits::chalk::fn_def_variance_query)]
    fn fn_def_variance(&self, krate: CrateId, fn_def_id: FnDefId) -> chalk::Variances;

    #[salsa::invoke(crate::traits::chalk::adt_variance_query)]
    fn adt_variance(&self, krate: CrateId, adt_id: chalk::AdtId) -> chalk::Variances;

    #[salsa::invoke(crate::traits::chalk::associated_ty_value_query)]
    fn associated_ty_value(
        &self,
        krate: CrateId,
        id: chalk::AssociatedTyValueId,
    ) -> Arc<chalk::AssociatedTyValue>;

    #[salsa::invoke(crate::traits::trait_solve_query)]
    fn trait_solve(
        &self,
        krate: CrateId,
        goal: crate::Canonical<crate::InEnvironment<crate::DomainGoal>>,
    ) -> Option<crate::traits::Solution>;

    #[salsa::invoke(crate::traits::chalk::program_clauses_for_chalk_env_query)]
    fn program_clauses_for_chalk_env(
        &self,
        krate: CrateId,
        env: chalk_ir::Environment<chalk::Interner>,
    ) -> chalk_ir::ProgramClauses<chalk::Interner>;
}

fn infer_wait(db: &dyn HirDatabase, def: DefWithBodyId) -> Arc<InferenceResult> {
    let _p = profile::span("infer:wait").detail(|| match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).name.to_string(),
        DefWithBodyId::StaticId(it) => {
            db.static_data(it).name.clone().unwrap_or_else(Name::missing).to_string()
        }
        DefWithBodyId::ConstId(it) => {
            db.const_data(it).name.clone().unwrap_or_else(Name::missing).to_string()
        }
    });
    db.infer_query(def)
}

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedTypeParamId(salsa::InternId);
impl_intern_key!(InternedTypeParamId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedOpaqueTyId(salsa::InternId);
impl_intern_key!(InternedOpaqueTyId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedClosureId(salsa::InternId);
impl_intern_key!(InternedClosureId);

/// This exists just for Chalk, because Chalk just has a single `FnDefId` where
/// we have different IDs for struct and enum variant constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct InternedCallableDefId(salsa::InternId);
impl_intern_key!(InternedCallableDefId);
