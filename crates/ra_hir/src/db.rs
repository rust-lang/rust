//! FIXME: write short doc here

use std::sync::Arc;

use hir_def::{GenericDefId, LocalStructFieldId, TraitId, VariantId};
use ra_arena::map::ArenaMap;
use ra_db::{salsa, CrateId};

use crate::{
    ty::{
        method_resolution::CrateImplBlocks,
        traits::{AssocTyValue, Impl},
        CallableDef, FnSig, GenericPredicate, InferenceResult, Substs, Ty, TyDefId, TypeCtor,
        ValueTyDefId,
    },
    DefWithBody, ImplBlock,
};

pub use hir_def::db::{
    BodyQuery, BodyWithSourceMapQuery, ConstDataQuery, CrateDefMapQuery, CrateLangItemsQuery,
    DefDatabase, DefDatabaseStorage, DocumentationQuery, EnumDataQuery, ExprScopesQuery,
    FunctionDataQuery, GenericParamsQuery, ImplDataQuery, InternDatabase, InternDatabaseStorage,
    LangItemQuery, ModuleLangItemsQuery, RawItemsQuery, RawItemsWithSourceMapQuery,
    StaticDataQuery, StructDataQuery, TraitDataQuery, TypeAliasDataQuery,
};
pub use hir_expand::db::{
    AstDatabase, AstDatabaseStorage, AstIdMapQuery, MacroArgQuery, MacroDefQuery, MacroExpandQuery,
    ParseMacroQuery,
};

#[salsa::query_group(HirDatabaseStorage)]
#[salsa::requires(salsa::Database)]
pub trait HirDatabase: DefDatabase {
    #[salsa::invoke(crate::ty::infer_query)]
    fn infer(&self, def: DefWithBody) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::ty::ty_query)]
    fn ty(&self, def: TyDefId) -> Ty;

    #[salsa::invoke(crate::ty::value_ty_query)]
    fn value_ty(&self, def: ValueTyDefId) -> Ty;

    #[salsa::invoke(crate::ty::field_types_query)]
    fn field_types(&self, var: VariantId) -> Arc<ArenaMap<LocalStructFieldId, Ty>>;

    #[salsa::invoke(crate::ty::callable_item_sig)]
    fn callable_item_signature(&self, def: CallableDef) -> FnSig;

    #[salsa::invoke(crate::ty::generic_predicates_for_param_query)]
    fn generic_predicates_for_param(
        &self,
        def: GenericDefId,
        param_idx: u32,
    ) -> Arc<[GenericPredicate]>;

    #[salsa::invoke(crate::ty::generic_predicates_query)]
    fn generic_predicates(&self, def: GenericDefId) -> Arc<[GenericPredicate]>;

    #[salsa::invoke(crate::ty::generic_defaults_query)]
    fn generic_defaults(&self, def: GenericDefId) -> Substs;

    #[salsa::invoke(crate::ty::method_resolution::CrateImplBlocks::impls_in_crate_query)]
    fn impls_in_crate(&self, krate: CrateId) -> Arc<CrateImplBlocks>;

    #[salsa::invoke(crate::ty::traits::impls_for_trait_query)]
    fn impls_for_trait(&self, krate: CrateId, trait_: TraitId) -> Arc<[ImplBlock]>;

    /// This provides the Chalk trait solver instance. Because Chalk always
    /// works from a specific crate, this query is keyed on the crate; and
    /// because Chalk does its own internal caching, the solver is wrapped in a
    /// Mutex and the query does an untracked read internally, to make sure the
    /// cached state is thrown away when input facts change.
    #[salsa::invoke(crate::ty::traits::trait_solver_query)]
    fn trait_solver(&self, krate: CrateId) -> crate::ty::traits::TraitSolver;

    // Interned IDs for Chalk integration
    #[salsa::interned]
    fn intern_type_ctor(&self, type_ctor: TypeCtor) -> crate::ty::TypeCtorId;
    #[salsa::interned]
    fn intern_chalk_impl(&self, impl_: Impl) -> crate::ty::traits::GlobalImplId;
    #[salsa::interned]
    fn intern_assoc_ty_value(
        &self,
        assoc_ty_value: AssocTyValue,
    ) -> crate::ty::traits::AssocTyValueId;

    #[salsa::invoke(crate::ty::traits::chalk::associated_ty_data_query)]
    fn associated_ty_data(
        &self,
        id: chalk_ir::TypeId,
    ) -> Arc<chalk_rust_ir::AssociatedTyDatum<chalk_ir::family::ChalkIr>>;

    #[salsa::invoke(crate::ty::traits::chalk::trait_datum_query)]
    fn trait_datum(
        &self,
        krate: CrateId,
        trait_id: chalk_ir::TraitId,
    ) -> Arc<chalk_rust_ir::TraitDatum<chalk_ir::family::ChalkIr>>;

    #[salsa::invoke(crate::ty::traits::chalk::struct_datum_query)]
    fn struct_datum(
        &self,
        krate: CrateId,
        struct_id: chalk_ir::StructId,
    ) -> Arc<chalk_rust_ir::StructDatum<chalk_ir::family::ChalkIr>>;

    #[salsa::invoke(crate::ty::traits::chalk::impl_datum_query)]
    fn impl_datum(
        &self,
        krate: CrateId,
        impl_id: chalk_ir::ImplId,
    ) -> Arc<chalk_rust_ir::ImplDatum<chalk_ir::family::ChalkIr>>;

    #[salsa::invoke(crate::ty::traits::chalk::associated_ty_value_query)]
    fn associated_ty_value(
        &self,
        krate: CrateId,
        id: chalk_rust_ir::AssociatedTyValueId,
    ) -> Arc<chalk_rust_ir::AssociatedTyValue<chalk_ir::family::ChalkIr>>;

    #[salsa::invoke(crate::ty::traits::trait_solve_query)]
    fn trait_solve(
        &self,
        krate: CrateId,
        goal: crate::ty::Canonical<crate::ty::InEnvironment<crate::ty::Obligation>>,
    ) -> Option<crate::ty::traits::Solution>;
}

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
