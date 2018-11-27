use std::sync::Arc;

use ra_syntax::{
    SyntaxNode,
    ast::FnDefNode,
};

use crate::{
    FileId,
    db::SyntaxDatabase,
    hir::function::{FnId, FnScopes},
    hir::module::{
        ModuleId, ModuleTree, ModuleSource,
        nameres::{ItemMap, InputModuleItems, FileItems, FileItemId}
    },
    input::SourceRootId,
    loc2id::{IdDatabase},
    Cancelable,
};

salsa::query_group! {
pub(crate) trait HirDatabase: SyntaxDatabase + IdDatabase {
        fn fn_scopes(fn_id: FnId) -> Arc<FnScopes> {
            type FnScopesQuery;
            use fn crate::hir::function::imp::fn_scopes;
        }

        fn file_items(file_id: FileId) -> Arc<FileItems> {
            type FileItemsQuery;
            storage dependencies;
            use fn crate::hir::module::nameres::file_items;
        }

        fn file_item(file_id: FileId, file_item_id: FileItemId) -> SyntaxNode {
            type FileItemQuery;
            storage dependencies;
            use fn crate::hir::module::nameres::file_item;
        }

        fn input_module_items(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<InputModuleItems>> {
            type InputModuleItemsQuery;
            use fn crate::hir::module::nameres::input_module_items;
        }
        fn item_map(source_root_id: SourceRootId) -> Cancelable<Arc<ItemMap>> {
            type ItemMapQuery;
            use fn crate::hir::module::nameres::item_map;
        }
        fn module_tree(source_root_id: SourceRootId) -> Cancelable<Arc<ModuleTree>> {
            type ModuleTreeQuery;
            use fn crate::hir::module::imp::module_tree;
        }
        fn fn_syntax(fn_id: FnId) -> FnDefNode {
            type FnSyntaxQuery;
            // Don't retain syntax trees in memory
            storage dependencies;
            use fn crate::hir::function::imp::fn_syntax;
        }
        fn submodules(source: ModuleSource) -> Cancelable<Arc<Vec<crate::hir::module::imp::Submodule>>> {
            type SubmodulesQuery;
            use fn crate::hir::module::imp::submodules;
        }
    }
}
