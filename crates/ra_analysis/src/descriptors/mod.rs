pub(crate) mod module;
pub(crate) mod function;

use std::sync::Arc;

use ra_syntax::{
    SmolStr,
    ast::{self, AstNode, FnDefNode},
    TextRange
};

use crate::{
    FileId, Cancelable,
    db::SyntaxDatabase,
    descriptors::module::{ModuleTree, ModuleId, ModuleScope},
    descriptors::function::{FnId, FnScopes, resolve_local_name},
    input::SourceRootId,
    syntax_ptr::{SyntaxPtrDatabase, LocalSyntaxPtr},
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

#[derive(Debug)]
pub struct ReferenceDescriptor {
    pub range: TextRange,
    pub name: String
}

#[derive(Debug)]
pub struct DeclarationDescriptor<'a> {
    pat: ast::BindPat<'a>,
    pub range: TextRange
}

impl<'a> DeclarationDescriptor<'a> {
    pub fn new(pat: ast::BindPat) -> DeclarationDescriptor {
        let range = pat.syntax().range();

        DeclarationDescriptor {
            pat,
            range
        }
    }

    pub fn find_all_refs(&self) -> Vec<ReferenceDescriptor> {
        let name_ptr = LocalSyntaxPtr::new(self.pat.syntax());

        let fn_def = match self.pat.syntax().ancestors().find_map(ast::FnDef::cast) {
            Some(def) => def,
            None => return Default::default()
        };

        let fn_scopes = FnScopes::new(fn_def);

        let refs : Vec<_> = fn_def.syntax().descendants()
            .filter_map(ast::NameRef::cast)
            .filter(|name_ref| {
                match resolve_local_name(*name_ref, &fn_scopes) {
                    None => false,
                    Some(entry) => entry.ptr() == name_ptr,
                }
            })
            .map(|name_ref| ReferenceDescriptor {
                name: name_ref.syntax().text().to_string(),
                range : name_ref.syntax().range(),
            })
            .collect();

        refs
    }
}
