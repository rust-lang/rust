use std::sync::Arc;

use parking_lot::Mutex;
use ra_syntax::{SyntaxNode, TreeArc, SmolStr, ast};
use ra_db::{SourceDatabase, salsa};

use crate::{
    HirFileId, MacroDefId, AstIdMap, ErasedFileAstId, Crate, Module, MacroCallLoc,
    Function, FnSignature, ExprScopes, TypeAlias,
    Struct, Enum, StructField,
    Const, ConstSignature, Static,
    DefWithBody, Trait,
    ids,
    nameres::{Namespace, ImportSourceMap, RawItems, CrateDefMap},
    ty::{InferenceResult, Ty, method_resolution::CrateImplBlocks, TypableDef, CallableDef, FnSig, TypeCtor, GenericPredicate, Substs},
    adt::{StructData, EnumData},
    impl_block::{ModuleImplBlocks, ImplSourceMap, ImplBlock},
    generics::{GenericParams, GenericDef},
    type_ref::TypeRef,
    traits::TraitData,
    lang_item::{LangItems, LangItemTarget},
};

// This database has access to source code, so queries here are not really
// incremental.
#[salsa::query_group(AstDatabaseStorage)]
pub trait AstDatabase: SourceDatabase {
    #[salsa::interned]
    fn intern_macro(&self, macro_call: MacroCallLoc) -> ids::MacroCallId;

    #[salsa::invoke(crate::source_id::AstIdMap::ast_id_map_query)]
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;
    #[salsa::transparent]
    #[salsa::invoke(crate::source_id::AstIdMap::file_item_query)]
    fn ast_id_to_node(&self, file_id: HirFileId, ast_id: ErasedFileAstId) -> TreeArc<SyntaxNode>;
    #[salsa::invoke(crate::ids::HirFileId::parse_or_expand_query)]
    fn parse_or_expand(&self, file_id: HirFileId) -> Option<TreeArc<SyntaxNode>>;

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
pub trait DefDatabase: SourceDatabase {
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

    #[salsa::invoke(crate::adt::StructData::struct_data_query)]
    fn struct_data(&self, s: Struct) -> Arc<StructData>;

    #[salsa::invoke(crate::adt::EnumData::enum_data_query)]
    fn enum_data(&self, e: Enum) -> Arc<EnumData>;

    #[salsa::invoke(crate::traits::TraitData::trait_data_query)]
    fn trait_data(&self, t: Trait) -> Arc<TraitData>;

    #[salsa::invoke(crate::traits::TraitItemsIndex::trait_items_index)]
    fn trait_items_index(&self, module: Module) -> crate::traits::TraitItemsIndex;

    #[salsa::invoke(RawItems::raw_items_query)]
    fn raw_items(&self, file_id: HirFileId) -> Arc<RawItems>;

    #[salsa::invoke(RawItems::raw_items_with_source_map_query)]
    fn raw_items_with_source_map(
        &self,
        file_id: HirFileId,
    ) -> (Arc<RawItems>, Arc<ImportSourceMap>);

    #[salsa::invoke(CrateDefMap::crate_def_map_query)]
    fn crate_def_map(&self, krate: Crate) -> Arc<CrateDefMap>;

    #[salsa::invoke(crate::impl_block::impls_in_module)]
    fn impls_in_module(&self, module: Module) -> Arc<ModuleImplBlocks>;

    #[salsa::invoke(crate::impl_block::impls_in_module_with_source_map_query)]
    fn impls_in_module_with_source_map(
        &self,
        module: Module,
    ) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>);

    #[salsa::invoke(crate::generics::GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDef) -> Arc<GenericParams>;

    #[salsa::invoke(crate::FnSignature::fn_signature_query)]
    fn fn_signature(&self, func: Function) -> Arc<FnSignature>;

    #[salsa::invoke(crate::type_alias::type_alias_ref_query)]
    fn type_alias_ref(&self, typ: TypeAlias) -> Arc<TypeRef>;

    #[salsa::invoke(crate::ConstSignature::const_signature_query)]
    fn const_signature(&self, konst: Const) -> Arc<ConstSignature>;

    #[salsa::invoke(crate::ConstSignature::static_signature_query)]
    fn static_signature(&self, konst: Static) -> Arc<ConstSignature>;

    #[salsa::invoke(crate::lang_item::LangItems::lang_items_query)]
    fn lang_items(&self, krate: Crate) -> Arc<LangItems>;

    #[salsa::invoke(crate::lang_item::LangItems::lang_item_query)]
    fn lang_item(&self, start_crate: Crate, item: SmolStr) -> Option<LangItemTarget>;
}

#[salsa::query_group(HirDatabaseStorage)]
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

    #[salsa::invoke(crate::ty::generic_predicates)]
    fn generic_predicates(&self, def: GenericDef) -> Arc<[GenericPredicate]>;

    #[salsa::invoke(crate::ty::generic_defaults)]
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
    /// Mutex and the query is marked volatile, to make sure the cached state is
    /// thrown away when input facts change.
    #[salsa::invoke(crate::ty::traits::solver_query)]
    #[salsa::volatile]
    fn solver(&self, krate: Crate) -> Arc<Mutex<crate::ty::traits::Solver>>;

    #[salsa::invoke(crate::ty::traits::implements_query)]
    fn implements(
        &self,
        krate: Crate,
        goal: crate::ty::Canonical<crate::ty::TraitRef>,
    ) -> Option<crate::ty::traits::Solution>;
}

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
