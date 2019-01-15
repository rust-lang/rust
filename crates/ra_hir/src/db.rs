use std::sync::Arc;

use ra_syntax::{SyntaxNode, TreeArc, SourceFile};
use ra_db::{SourceRootId, LocationIntener, SyntaxDatabase, Cancelable};

use crate::{
    DefLoc, DefId, MacroCallLoc, MacroCallId, Name, HirFileId,
    SourceFileItems, SourceItemId, Crate,
    query_definitions,
    FnSignature, FnScopes,
    macros::MacroExpansion,
    module_tree::{ModuleId, ModuleTree},
    nameres::{ItemMap, InputModuleItems},
    ty::{InferenceResult, Ty, method_resolution::CrateImplBlocks},
    adt::{StructData, EnumData, EnumVariantData},
    impl_block::ModuleImplBlocks,
};

salsa::query_group! {

pub trait HirDatabase: SyntaxDatabase
    + AsRef<LocationIntener<DefLoc, DefId>>
    + AsRef<LocationIntener<MacroCallLoc, MacroCallId>>
{
    fn hir_source_file(file_id: HirFileId) -> TreeArc<SourceFile> {
        type HirSourceFileQuery;
        use fn HirFileId::hir_source_file;
    }

    fn expand_macro_invocation(invoc: MacroCallId) -> Option<Arc<MacroExpansion>> {
        type ExpandMacroCallQuery;
        use fn crate::macros::expand_macro_invocation;
    }

    fn fn_scopes(def_id: DefId) -> Arc<FnScopes> {
        type FnScopesQuery;
        use fn query_definitions::fn_scopes;
    }

    fn struct_data(def_id: DefId) -> Arc<StructData> {
        type StructDataQuery;
        use fn crate::adt::StructData::struct_data_query;
    }

    fn enum_data(def_id: DefId) -> Arc<EnumData> {
        type EnumDataQuery;
        use fn crate::adt::EnumData::enum_data_query;
    }

    fn enum_variant_data(def_id: DefId) -> Arc<EnumVariantData> {
        type EnumVariantDataQuery;
        use fn crate::adt::EnumVariantData::enum_variant_data_query;
    }

    fn infer(def_id: DefId) -> Cancelable<Arc<InferenceResult>> {
        type InferQuery;
        use fn crate::ty::infer;
    }

    fn type_for_def(def_id: DefId) -> Cancelable<Ty> {
        type TypeForDefQuery;
        use fn crate::ty::type_for_def;
    }

    fn type_for_field(def_id: DefId, field: Name) -> Cancelable<Option<Ty>> {
        type TypeForFieldQuery;
        use fn crate::ty::type_for_field;
    }

    fn file_items(file_id: HirFileId) -> Arc<SourceFileItems> {
        type SourceFileItemsQuery;
        use fn query_definitions::file_items;
    }

    fn file_item(source_item_id: SourceItemId) -> TreeArc<SyntaxNode> {
        type FileItemQuery;
        use fn query_definitions::file_item;
    }

    fn submodules(source: SourceItemId) -> Arc<Vec<crate::module_tree::Submodule>> {
        type SubmodulesQuery;
        use fn crate::module_tree::Submodule::submodules_query;
    }

    fn input_module_items(source_root_id: SourceRootId, module_id: ModuleId) -> Arc<InputModuleItems> {
        type InputModuleItemsQuery;
        use fn query_definitions::input_module_items;
    }

    fn item_map(source_root_id: SourceRootId) -> Arc<ItemMap> {
        type ItemMapQuery;
        use fn query_definitions::item_map;
    }

    fn module_tree(source_root_id: SourceRootId) -> Arc<ModuleTree> {
        type ModuleTreeQuery;
        use fn crate::module_tree::ModuleTree::module_tree_query;
    }

    fn impls_in_module(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<ModuleImplBlocks>> {
        type ImplsInModuleQuery;
        use fn crate::impl_block::impls_in_module;
    }

    fn impls_in_crate(krate: Crate) -> Cancelable<Arc<CrateImplBlocks>> {
        type ImplsInCrateQuery;
        use fn crate::ty::method_resolution::CrateImplBlocks::impls_in_crate_query;
    }

    fn body_hir(def_id: DefId) -> Arc<crate::expr::Body> {
        type BodyHirQuery;
        use fn crate::expr::body_hir;
    }

    fn body_syntax_mapping(def_id: DefId) -> Arc<crate::expr::BodySyntaxMapping> {
        type BodySyntaxMappingQuery;
        use fn crate::expr::body_syntax_mapping;
    }

    fn fn_signature(def_id: DefId) -> Arc<FnSignature> {
        type FnSignatureQuery;
        use fn crate::FnSignature::fn_signature_query;
    }
}

}
