use std::sync::Arc;

use ra_db::{salsa, SourceDatabase};
use ra_syntax::{ast, Parse, SmolStr, SyntaxNode};

use crate::{
    adt::{EnumData, StructData},
    generics::{GenericDef, GenericParams},
    ids,
    impl_block::{ImplBlock, ImplSourceMap, ModuleImplBlocks},
    lang_item::{LangItemTarget, LangItems},
    nameres::{CrateDefMap, ImportSourceMap, Namespace, RawItems},
    traits::TraitData,
    ty::{
        method_resolution::CrateImplBlocks, CallableDef, FnSig, GenericPredicate, InferenceResult,
        Substs, Ty, TypableDef, TypeCtor,
    },
    type_alias::TypeAliasData,
    AstIdMap, Const, ConstData, Crate, DefWithBody, Enum, ErasedFileAstId, ExprScopes, FnData,
    Function, HirFileId, MacroCallLoc, MacroDefId, Module, Static, Struct, StructField, Trait,
    TypeAlias,
};

/// We store all interned things in the single QueryGroup.
///
/// This is done mainly to allow both "volatile" `AstDatabase` and "stable"
/// `DefDatabase` to access macros, without adding hard dependencies between the
/// two.
#[salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    #[salsa::interned]
    fn intern_macro(&self, macro_call: MacroCallLoc) -> ids::MacroCallId;
    #[salsa::interned]
    fn intern_function(&self, loc: ids::ItemLoc<ast::FnDef>) -> ids::FunctionId;
    #[salsa::interned]
    fn intern_struct(&self, loc: ids::ItemLoc<ast::StructDef>) -> ids::StructId;
    #[salsa::interned]
    fn intern_enum(&self, loc: ids::ItemLoc<ast::EnumDef>) -> ids::EnumId;
    #[salsa::interned]
    fn intern_const(&self, loc: ids::ItemLoc<ast::ConstDef>) -> ids::ConstId;
    #[salsa::interned]
    fn intern_static(&self, loc: ids::ItemLoc<ast::StaticDef>) -> ids::StaticId;
    #[salsa::interned]
    fn intern_trait(&self, loc: ids::ItemLoc<ast::TraitDef>) -> ids::TraitId;
    #[salsa::interned]
    fn intern_type_alias(&self, loc: ids::ItemLoc<ast::TypeAliasDef>) -> ids::TypeAliasId;

    // Interned IDs for Chalk integration
    #[salsa::interned]
    fn intern_type_ctor(&self, type_ctor: TypeCtor) -> ids::TypeCtorId;
    #[salsa::interned]
    fn intern_impl_block(&self, impl_block: ImplBlock) -> ids::GlobalImplId;
}

/// This database has access to source code, so queries here are not really
/// incremental.
#[salsa::query_group(AstDatabaseStorage)]
pub trait AstDatabase: InternDatabase {
    #[salsa::invoke(crate::source_id::AstIdMap::ast_id_map_query)]
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;

    #[salsa::transparent]
    #[salsa::invoke(crate::source_id::AstIdMap::file_item_query)]
    fn ast_id_to_node(&self, file_id: HirFileId, ast_id: ErasedFileAstId) -> SyntaxNode;

    #[salsa::transparent]
    #[salsa::invoke(crate::ids::HirFileId::parse_or_expand_query)]
    fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode>;

    #[salsa::invoke(crate::ids::HirFileId::parse_macro_query)]
    fn parse_macro(&self, macro_file: ids::MacroFile) -> Option<Parse<SyntaxNode>>;

    #[salsa::invoke(crate::ids::macro_def_query)]
    fn macro_def(&self, macro_id: MacroDefId) -> Option<Arc<mbe::MacroRules>>;

    #[salsa::invoke(crate::ids::macro_arg_query)]
    fn macro_arg(&self, macro_call: ids::MacroCallId) -> Option<Arc<tt::Subtree>>;

    #[salsa::invoke(crate::ids::macro_expand_query)]
    fn macro_expand(&self, macro_call: ids::MacroCallId) -> Result<Arc<tt::Subtree>, String>;
}

// This database uses `AstDatabase` internally,
#[salsa::query_group(DefDatabaseStorage)]
#[salsa::requires(AstDatabase)]
pub trait DefDatabase: InternDatabase {
    #[salsa::invoke(crate::adt::StructData::struct_data_query)]
    fn struct_data(&self, s: Struct) -> Arc<StructData>;

    #[salsa::invoke(crate::adt::EnumData::enum_data_query)]
    fn enum_data(&self, e: Enum) -> Arc<EnumData>;

    #[salsa::invoke(crate::traits::TraitData::trait_data_query)]
    fn trait_data(&self, t: Trait) -> Arc<TraitData>;

    #[salsa::invoke(crate::traits::TraitItemsIndex::trait_items_index)]
    fn trait_items_index(&self, module: Module) -> crate::traits::TraitItemsIndex;

    #[salsa::invoke(RawItems::raw_items_with_source_map_query)]
    fn raw_items_with_source_map(
        &self,
        file_id: HirFileId,
    ) -> (Arc<RawItems>, Arc<ImportSourceMap>);

    #[salsa::invoke(RawItems::raw_items_query)]
    fn raw_items(&self, file_id: HirFileId) -> Arc<RawItems>;

    #[salsa::invoke(CrateDefMap::crate_def_map_query)]
    fn crate_def_map(&self, krate: Crate) -> Arc<CrateDefMap>;

    #[salsa::invoke(crate::impl_block::impls_in_module_with_source_map_query)]
    fn impls_in_module_with_source_map(
        &self,
        module: Module,
    ) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>);

    #[salsa::invoke(crate::impl_block::impls_in_module)]
    fn impls_in_module(&self, module: Module) -> Arc<ModuleImplBlocks>;

    #[salsa::invoke(crate::generics::GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDef) -> Arc<GenericParams>;

    #[salsa::invoke(crate::FnData::fn_data_query)]
    fn fn_data(&self, func: Function) -> Arc<FnData>;

    #[salsa::invoke(crate::type_alias::type_alias_data_query)]
    fn type_alias_data(&self, typ: TypeAlias) -> Arc<TypeAliasData>;

    #[salsa::invoke(crate::ConstData::const_data_query)]
    fn const_data(&self, konst: Const) -> Arc<ConstData>;

    #[salsa::invoke(crate::ConstData::static_data_query)]
    fn static_data(&self, konst: Static) -> Arc<ConstData>;

    #[salsa::invoke(crate::lang_item::LangItems::module_lang_items_query)]
    fn module_lang_items(&self, module: Module) -> Option<Arc<LangItems>>;

    #[salsa::invoke(crate::lang_item::LangItems::lang_items_query)]
    fn lang_items(&self, krate: Crate) -> Arc<LangItems>;

    #[salsa::invoke(crate::lang_item::LangItems::lang_item_query)]
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
