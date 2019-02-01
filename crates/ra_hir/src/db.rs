use std::sync::Arc;

use ra_syntax::{SyntaxNode, TreeArc, SourceFile};
use ra_db::{SourceDatabase, salsa};

use crate::{
    MacroCallId, HirFileId,
    SourceFileItems, SourceItemId, Crate, Module, HirInterner,
    query_definitions,
    Function, FnSignature, ExprScopes,
    Struct, Enum, StructField,
    macros::MacroExpansion,
    module_tree::ModuleTree,
    nameres::{ItemMap, lower::{LoweredModule, ImportSourceMap}},
    ty::{InferenceResult, Ty, method_resolution::CrateImplBlocks, TypableDef},
    adt::{StructData, EnumData},
    impl_block::{ModuleImplBlocks, ImplSourceMap},
    generics::{GenericParams, GenericDef},
    ids::SourceFileItemId,
};

#[salsa::query_group(PersistentHirDatabaseStorage)]
pub trait PersistentHirDatabase: SourceDatabase + AsRef<HirInterner> {
    #[salsa::invoke(HirFileId::hir_parse)]
    fn hir_parse(&self, file_id: HirFileId) -> TreeArc<SourceFile>;

    #[salsa::invoke(crate::macros::expand_macro_invocation)]
    fn expand_macro_invocation(&self, invoc: MacroCallId) -> Option<Arc<MacroExpansion>>;

    #[salsa::invoke(crate::adt::StructData::struct_data_query)]
    fn struct_data(&self, s: Struct) -> Arc<StructData>;

    #[salsa::invoke(crate::adt::EnumData::enum_data_query)]
    fn enum_data(&self, e: Enum) -> Arc<EnumData>;

    #[salsa::invoke(query_definitions::file_items)]
    fn file_items(&self, file_id: HirFileId) -> Arc<SourceFileItems>;

    #[salsa::invoke(query_definitions::file_item)]
    fn file_item(&self, source_item_id: SourceItemId) -> TreeArc<SyntaxNode>;

    #[salsa::invoke(crate::module_tree::Submodule::submodules_query)]
    fn submodules(
        &self,
        file_id: HirFileId,
        delc_id: Option<SourceFileItemId>,
    ) -> Arc<Vec<crate::module_tree::Submodule>>;

    #[salsa::invoke(crate::nameres::lower::LoweredModule::lower_module_query)]
    fn lower_module(&self, module: Module) -> (Arc<LoweredModule>, Arc<ImportSourceMap>);

    #[salsa::invoke(crate::nameres::lower::LoweredModule::lower_module_module_query)]
    fn lower_module_module(&self, module: Module) -> Arc<LoweredModule>;

    #[salsa::invoke(crate::nameres::lower::LoweredModule::lower_module_source_map_query)]
    fn lower_module_source_map(&self, module: Module) -> Arc<ImportSourceMap>;

    #[salsa::invoke(crate::nameres::ItemMap::item_map_query)]
    fn item_map(&self, krate: Crate) -> Arc<ItemMap>;

    #[salsa::invoke(crate::module_tree::ModuleTree::module_tree_query)]
    fn module_tree(&self, krate: Crate) -> Arc<ModuleTree>;

    #[salsa::invoke(crate::impl_block::impls_in_module)]
    fn impls_in_module(&self, module: Module) -> Arc<ModuleImplBlocks>;

    #[salsa::invoke(crate::impl_block::impls_in_module_source_map_query)]
    fn impls_in_module_source_map(&self, module: Module) -> Arc<ImplSourceMap>;

    #[salsa::invoke(crate::impl_block::impls_in_module_with_source_map_query)]
    fn impls_in_module_with_source_map(
        &self,
        module: Module,
    ) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>);

    #[salsa::invoke(crate::generics::GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDef) -> Arc<GenericParams>;

    #[salsa::invoke(crate::FnSignature::fn_signature_query)]
    fn fn_signature(&self, func: Function) -> Arc<FnSignature>;
}

#[salsa::query_group(HirDatabaseStorage)]
pub trait HirDatabase: PersistentHirDatabase {
    #[salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, func: Function) -> Arc<ExprScopes>;

    #[salsa::invoke(crate::ty::infer)]
    fn infer(&self, func: Function) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::ty::type_for_def)]
    fn type_for_def(&self, def: TypableDef) -> Ty;

    #[salsa::invoke(crate::ty::type_for_field)]
    fn type_for_field(&self, field: StructField) -> Ty;

    #[salsa::invoke(crate::expr::body_hir)]
    fn body_hir(&self, func: Function) -> Arc<crate::expr::Body>;

    #[salsa::invoke(crate::expr::body_syntax_mapping)]
    fn body_syntax_mapping(&self, func: Function) -> Arc<crate::expr::BodySyntaxMapping>;

    #[salsa::invoke(crate::ty::method_resolution::CrateImplBlocks::impls_in_crate_query)]
    fn impls_in_crate(&self, krate: Crate) -> Arc<CrateImplBlocks>;
}
