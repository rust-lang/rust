use std::sync::Arc;

use ra_syntax::{SyntaxNode, SourceFileNode};
use ra_db::{SourceRootId, LocationIntener, SyntaxDatabase, Cancelable};

use crate::{
    DefLoc, DefId, MacroCallLoc, MacroCallId, Name, HirFileId,
    SourceFileItems, SourceItemId,
    query_definitions,
    FnScopes,
    macros::MacroExpansion,
    module::{ModuleId, ModuleTree, ModuleSource,
    nameres::{ItemMap, InputModuleItems}},
    ty::{InferenceResult, Ty},
    adt::{StructData, EnumData},
    impl_block::ModuleImplBlocks,
};

salsa::query_group! {

pub trait HirDatabase: SyntaxDatabase
    + AsRef<LocationIntener<DefLoc, DefId>>
    + AsRef<LocationIntener<MacroCallLoc, MacroCallId>>
{
    fn hir_source_file(file_id: HirFileId) -> SourceFileNode {
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

    fn struct_data(def_id: DefId) -> Cancelable<Arc<StructData>> {
        type StructDataQuery;
        use fn query_definitions::struct_data;
    }

    fn enum_data(def_id: DefId) -> Cancelable<Arc<EnumData>> {
        type EnumDataQuery;
        use fn query_definitions::enum_data;
    }

    fn infer(def_id: DefId) -> Cancelable<Arc<InferenceResult>> {
        type InferQuery;
        use fn crate::ty::infer;
    }

    fn type_for_def(def_id: DefId) -> Cancelable<Ty> {
        type TypeForDefQuery;
        use fn crate::ty::type_for_def;
    }

    fn type_for_field(def_id: DefId, field: Name) -> Cancelable<Ty> {
        type TypeForFieldQuery;
        use fn crate::ty::type_for_field;
    }

    fn file_items(file_id: HirFileId) -> Arc<SourceFileItems> {
        type SourceFileItemsQuery;
        use fn query_definitions::file_items;
    }

    fn file_item(source_item_id: SourceItemId) -> SyntaxNode {
        type FileItemQuery;
        use fn query_definitions::file_item;
    }

    fn submodules(source: ModuleSource) -> Cancelable<Arc<Vec<crate::module::imp::Submodule>>> {
        type SubmodulesQuery;
        use fn query_definitions::submodules;
    }

    fn input_module_items(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<InputModuleItems>> {
        type InputModuleItemsQuery;
        use fn query_definitions::input_module_items;
    }
    fn item_map(source_root_id: SourceRootId) -> Cancelable<Arc<ItemMap>> {
        type ItemMapQuery;
        use fn query_definitions::item_map;
    }
    fn module_tree(source_root_id: SourceRootId) -> Cancelable<Arc<ModuleTree>> {
        type ModuleTreeQuery;
        use fn crate::module::imp::module_tree;
    }

    fn impls_in_module(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<ModuleImplBlocks>> {
        type ImplsInCrateQuery;
        use fn crate::impl_block::impls_in_module;
    }
}

}
