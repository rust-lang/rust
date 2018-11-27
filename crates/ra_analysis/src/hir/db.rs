use std::sync::Arc;

use ra_syntax::{
    SyntaxNode,
    ast::FnDefNode,
};

use crate::{
    FileId,
    db::SyntaxDatabase,
    hir::{
        SourceFileItems, SourceItemId,
        query_definitions,
        function::{FnScopes},
        module::{ModuleId, ModuleTree, ModuleSource,
        nameres::{ItemMap, InputModuleItems}},
    },
    input::SourceRootId,
    loc2id::{DefLoc, DefId, FnId, LocationIntener},
    Cancelable,
};

salsa::query_group! {

pub(crate) trait HirDatabase: SyntaxDatabase
    + AsRef<LocationIntener<DefLoc, DefId>>
    + AsRef<LocationIntener<SourceItemId, FnId>>
{
    fn fn_scopes(fn_id: FnId) -> Arc<FnScopes> {
        type FnScopesQuery;
        use fn query_definitions::fn_scopes;
    }
    fn fn_syntax(fn_id: FnId) -> FnDefNode {
        type FnSyntaxQuery;
        // Don't retain syntax trees in memory
        storage dependencies;
        use fn query_definitions::fn_syntax;
    }

    fn file_items(file_id: FileId) -> Arc<SourceFileItems> {
        type SourceFileItemsQuery;
        storage dependencies;
        use fn query_definitions::file_items;
    }

    fn file_item(source_item_id: SourceItemId) -> SyntaxNode {
        type FileItemQuery;
        storage dependencies;
        use fn query_definitions::file_item;
    }

    fn submodules(source: ModuleSource) -> Cancelable<Arc<Vec<crate::hir::module::imp::Submodule>>> {
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
        use fn crate::hir::module::imp::module_tree;
    }
}

}
