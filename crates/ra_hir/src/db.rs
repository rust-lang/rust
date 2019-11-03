//! FIXME: write short doc here

use std::sync::Arc;

use ra_db::salsa;
use ra_syntax::SmolStr;

use crate::{
    debug::HirDebugDatabase,
    generics::{GenericDef, GenericParams},
    ids,
    impl_block::{ImplBlock, ImplSourceMap, ModuleImplBlocks},
    lang_item::{LangItemTarget, LangItems},
    traits::TraitData,
    ty::{
        method_resolution::CrateImplBlocks, traits::Impl, CallableDef, FnSig, GenericPredicate,
        InferenceResult, Substs, Ty, TypableDef, TypeCtor,
    },
    type_alias::TypeAliasData,
    Const, ConstData, Crate, DefWithBody, ExprScopes, FnData, Function, Module, Namespace, Static,
    StructField, Trait, TypeAlias,
};

pub use hir_def::db::{
    CrateDefMapQuery, DefDatabase2, DefDatabase2Storage, EnumDataQuery, InternDatabase,
    InternDatabaseStorage, RawItemsQuery, RawItemsWithSourceMapQuery, StructDataQuery,
};
pub use hir_expand::db::{
    AstDatabase, AstDatabaseStorage, AstIdMapQuery, MacroArgQuery, MacroDefQuery, MacroExpandQuery,
    ParseMacroQuery,
};

// This database uses `AstDatabase` internally,
#[salsa::query_group(DefDatabaseStorage)]
#[salsa::requires(AstDatabase)]
pub trait DefDatabase: HirDebugDatabase + DefDatabase2 {
    #[salsa::invoke(crate::traits::TraitData::trait_data_query)]
    fn trait_data(&self, t: Trait) -> Arc<TraitData>;

    #[salsa::invoke(crate::traits::TraitItemsIndex::trait_items_index)]
    fn trait_items_index(&self, module: Module) -> crate::traits::TraitItemsIndex;

    #[salsa::invoke(ModuleImplBlocks::impls_in_module_with_source_map_query)]
    fn impls_in_module_with_source_map(
        &self,
        module: Module,
    ) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>);

    #[salsa::invoke(ModuleImplBlocks::impls_in_module_query)]
    fn impls_in_module(&self, module: Module) -> Arc<ModuleImplBlocks>;

    #[salsa::invoke(crate::generics::GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDef) -> Arc<GenericParams>;

    #[salsa::invoke(FnData::fn_data_query)]
    fn fn_data(&self, func: Function) -> Arc<FnData>;

    #[salsa::invoke(TypeAliasData::type_alias_data_query)]
    fn type_alias_data(&self, typ: TypeAlias) -> Arc<TypeAliasData>;

    #[salsa::invoke(ConstData::const_data_query)]
    fn const_data(&self, konst: Const) -> Arc<ConstData>;

    #[salsa::invoke(ConstData::static_data_query)]
    fn static_data(&self, konst: Static) -> Arc<ConstData>;

    #[salsa::invoke(LangItems::module_lang_items_query)]
    fn module_lang_items(&self, module: Module) -> Option<Arc<LangItems>>;

    #[salsa::invoke(LangItems::crate_lang_items_query)]
    fn crate_lang_items(&self, krate: Crate) -> Arc<LangItems>;

    #[salsa::invoke(LangItems::lang_item_query)]
    fn lang_item(&self, start_crate: Crate, item: SmolStr) -> Option<LangItemTarget>;

    #[salsa::invoke(crate::code_model::docs::documentation_query)]
    fn documentation(&self, def: crate::DocDef) -> Option<crate::Documentation>;
}

#[salsa::query_group(HirDatabaseStorage)]
#[salsa::requires(salsa::Database)]
pub trait HirDatabase: DefDatabase + AstDatabase {
    #[salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, def: DefWithBody) -> Arc<ExprScopes>;

    #[salsa::invoke(crate::ty::infer_query)]
    fn infer(&self, def: DefWithBody) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::ty::type_for_def)]
    fn type_for_def(&self, def: TypableDef, ns: Namespace) -> Ty;

    #[salsa::invoke(crate::ty::type_for_field)]
    fn type_for_field(&self, field: StructField) -> Ty;

    #[salsa::invoke(crate::ty::callable_item_sig)]
    fn callable_item_signature(&self, def: CallableDef) -> FnSig;

    #[salsa::invoke(crate::ty::generic_predicates_for_param_query)]
    fn generic_predicates_for_param(
        &self,
        def: GenericDef,
        param_idx: u32,
    ) -> Arc<[GenericPredicate]>;

    #[salsa::invoke(crate::ty::generic_predicates_query)]
    fn generic_predicates(&self, def: GenericDef) -> Arc<[GenericPredicate]>;

    #[salsa::invoke(crate::ty::generic_defaults_query)]
    fn generic_defaults(&self, def: GenericDef) -> Substs;

    #[salsa::invoke(crate::expr::body_with_source_map_query)]
    fn body_with_source_map(
        &self,
        def: DefWithBody,
    ) -> (Arc<crate::expr::Body>, Arc<crate::expr::BodySourceMap>);

    #[salsa::invoke(crate::expr::body_hir_query)]
    fn body_hir(&self, def: DefWithBody) -> Arc<crate::expr::Body>;

    #[salsa::invoke(crate::ty::method_resolution::CrateImplBlocks::impls_in_crate_query)]
    fn impls_in_crate(&self, krate: Crate) -> Arc<CrateImplBlocks>;

    #[salsa::invoke(crate::ty::traits::impls_for_trait_query)]
    fn impls_for_trait(&self, krate: Crate, trait_: Trait) -> Arc<[ImplBlock]>;

    /// This provides the Chalk trait solver instance. Because Chalk always
    /// works from a specific crate, this query is keyed on the crate; and
    /// because Chalk does its own internal caching, the solver is wrapped in a
    /// Mutex and the query does an untracked read internally, to make sure the
    /// cached state is thrown away when input facts change.
    #[salsa::invoke(crate::ty::traits::trait_solver_query)]
    fn trait_solver(&self, krate: Crate) -> crate::ty::traits::TraitSolver;

    // Interned IDs for Chalk integration
    #[salsa::interned]
    fn intern_type_ctor(&self, type_ctor: TypeCtor) -> ids::TypeCtorId;
    #[salsa::interned]
    fn intern_impl(&self, impl_: Impl) -> ids::GlobalImplId;

    #[salsa::invoke(crate::ty::traits::chalk::associated_ty_data_query)]
    fn associated_ty_data(&self, id: chalk_ir::TypeId) -> Arc<chalk_rust_ir::AssociatedTyDatum>;

    #[salsa::invoke(crate::ty::traits::chalk::trait_datum_query)]
    fn trait_datum(
        &self,
        krate: Crate,
        trait_id: chalk_ir::TraitId,
    ) -> Arc<chalk_rust_ir::TraitDatum>;

    #[salsa::invoke(crate::ty::traits::chalk::struct_datum_query)]
    fn struct_datum(
        &self,
        krate: Crate,
        struct_id: chalk_ir::StructId,
    ) -> Arc<chalk_rust_ir::StructDatum>;

    #[salsa::invoke(crate::ty::traits::chalk::impl_datum_query)]
    fn impl_datum(&self, krate: Crate, impl_id: chalk_ir::ImplId) -> Arc<chalk_rust_ir::ImplDatum>;

    #[salsa::invoke(crate::ty::traits::trait_solve_query)]
    fn trait_solve(
        &self,
        krate: Crate,
        goal: crate::ty::Canonical<crate::ty::InEnvironment<crate::ty::Obligation>>,
    ) -> Option<crate::ty::traits::Solution>;
}

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
