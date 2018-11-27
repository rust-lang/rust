use std::sync::Arc;

use ra_syntax::{
    SyntaxNode,
    ast::FnDefNode,
};

use crate::{
    FileId,
    db::SyntaxDatabase,
    hir::{FileItems, FileItemId},
    hir::query_definitions,
    hir::function::{FnId, FnScopes},
    hir::module::{
        ModuleId, ModuleTree, ModuleSource,
        nameres::{ItemMap, InputModuleItems}
    },
    input::SourceRootId,
    Cancelable,
};

salsa::query_group! {

pub(crate) trait HirDatabase: SyntaxDatabase {
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

    fn file_items(file_id: FileId) -> Arc<FileItems> {
        type FileItemsQuery;
        storage dependencies;
        use fn query_definitions::file_items;
    }

    fn file_item(file_id: FileId, file_item_id: FileItemId) -> SyntaxNode {
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
