pub(crate) mod module;
pub(crate) mod function;

use std::sync::Arc;

use ra_syntax::{
    SmolStr,
    ast::{FnDefNode},
};

use crate::{
    FileId, Cancelable,
    db::SyntaxDatabase,
    descriptors::module::{ModuleTree, ModuleId, ModuleScope},
    descriptors::function::{FnId, FnScopes},
    input::SourceRootId,
    syntax_ptr::SyntaxPtrDatabase,
};


salsa::query_group! {
    pub(crate) trait DescriptorDatabase: SyntaxDatabase + SyntaxPtrDatabase {
        fn module_tree(source_root_id: SourceRootId) -> Cancelable<Arc<ModuleTree>> {
            type ModuleTreeQuery;
            use fn module::imp::module_tree;
        }
        fn submodules(file_id: FileId) -> Cancelable<Arc<Vec<SmolStr>>> {
            type SubmodulesQuery;
            use fn module::imp::submodules;
        }
        fn module_scope(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<ModuleScope>> {
            type ModuleScopeQuery;
            use fn module::imp::module_scope;
        }
        fn fn_syntax(fn_id: FnId) -> FnDefNode {
            type FnSyntaxQuery;
            // Don't retain syntax trees in memory
            storage volatile;
            use fn function::imp::fn_syntax;
        }
        fn fn_scopes(fn_id: FnId) -> Arc<FnScopes> {
            type FnScopesQuery;
            use fn function::imp::fn_scopes;
        }
    }
}
