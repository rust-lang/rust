use std::sync::Arc;

use ra_syntax::{SyntaxNode, TreeArc, SourceFile};
use ra_db::{SourceRootId, LocationIntener, SyntaxDatabase, salsa};

use crate::{
    DefLoc, DefId, MacroCallLoc, MacroCallId, Name, HirFileId,
    SourceFileItems, SourceItemId, Crate,
    query_definitions,
    FnSignature, FnScopes,
    macros::MacroExpansion,
    module_tree::{ModuleId, ModuleTree},
    nameres::{ItemMap, lower::{LoweredModule, ImportSourceMap}},
    ty::{InferenceResult, Ty, method_resolution::CrateImplBlocks},
    adt::{StructData, EnumData, EnumVariantData},
    impl_block::ModuleImplBlocks,
    generics::GenericParams,
};

#[salsa::query_group]
pub trait HirDatabase:
    SyntaxDatabase
    + AsRef<LocationIntener<DefLoc, DefId>>
    + AsRef<LocationIntener<MacroCallLoc, MacroCallId>>
{
    #[salsa::invoke(HirFileId::hir_source_file)]
    fn hir_source_file(&self, file_id: HirFileId) -> TreeArc<SourceFile>;

    #[salsa::invoke(crate::macros::expand_macro_invocation)]
    fn expand_macro_invocation(&self, invoc: MacroCallId) -> Option<Arc<MacroExpansion>>;

    #[salsa::invoke(query_definitions::fn_scopes)]
    fn fn_scopes(&self, def_id: DefId) -> Arc<FnScopes>;

    #[salsa::invoke(crate::adt::StructData::struct_data_query)]
    fn struct_data(&self, def_id: DefId) -> Arc<StructData>;

    #[salsa::invoke(crate::adt::EnumData::enum_data_query)]
    fn enum_data(&self, def_id: DefId) -> Arc<EnumData>;

    #[salsa::invoke(crate::adt::EnumVariantData::enum_variant_data_query)]
    fn enum_variant_data(&self, def_id: DefId) -> Arc<EnumVariantData>;

    #[salsa::invoke(crate::ty::infer)]
    fn infer(&self, def_id: DefId) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::ty::type_for_def)]
    fn type_for_def(&self, def_id: DefId) -> Ty;

    #[salsa::invoke(crate::ty::type_for_field)]
    fn type_for_field(&self, def_id: DefId, field: Name) -> Option<Ty>;

    #[salsa::invoke(query_definitions::file_items)]
    fn file_items(&self, file_id: HirFileId) -> Arc<SourceFileItems>;

    #[salsa::invoke(query_definitions::file_item)]
    fn file_item(&self, source_item_id: SourceItemId) -> TreeArc<SyntaxNode>;

    #[salsa::invoke(crate::module_tree::Submodule::submodules_query)]
    fn submodules(&self, source: SourceItemId) -> Arc<Vec<crate::module_tree::Submodule>>;

    #[salsa::invoke(crate::nameres::lower::LoweredModule::lower_module_query)]
    fn lower_module(
        &self,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> (Arc<LoweredModule>, Arc<ImportSourceMap>);

    #[salsa::invoke(crate::nameres::lower::LoweredModule::lower_module_module_query)]
    fn lower_module_module(
        &self,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Arc<LoweredModule>;

    #[salsa::invoke(crate::nameres::lower::LoweredModule::lower_module_source_map_query)]
    fn lower_module_source_map(
        &self,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Arc<ImportSourceMap>;

    #[salsa::invoke(query_definitions::item_map)]
    fn item_map(&self, source_root_id: SourceRootId) -> Arc<ItemMap>;

    #[salsa::invoke(crate::module_tree::ModuleTree::module_tree_query)]
    fn module_tree(&self, source_root_id: SourceRootId) -> Arc<ModuleTree>;

    #[salsa::invoke(crate::impl_block::impls_in_module)]
    fn impls_in_module(
        &self,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Arc<ModuleImplBlocks>;

    #[salsa::invoke(crate::ty::method_resolution::CrateImplBlocks::impls_in_crate_query)]
    fn impls_in_crate(&self, krate: Crate) -> Arc<CrateImplBlocks>;

    #[salsa::invoke(crate::expr::body_hir)]
    fn body_hir(&self, def_id: DefId) -> Arc<crate::expr::Body>;

    #[salsa::invoke(crate::expr::body_syntax_mapping)]
    fn body_syntax_mapping(&self, def_id: DefId) -> Arc<crate::expr::BodySyntaxMapping>;

    #[salsa::invoke(crate::generics::GenericParams::generic_params_query)]
    fn generic_params(&self, def_id: DefId) -> Arc<GenericParams>;

    #[salsa::invoke(crate::FnSignature::fn_signature_query)]
    fn fn_signature(&self, def_id: DefId) -> Arc<FnSignature>;
}
