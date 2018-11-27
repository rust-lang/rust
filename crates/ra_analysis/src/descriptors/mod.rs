pub(crate) mod function;
pub(crate) mod module;
mod path;

use std::sync::Arc;

use ra_syntax::{
    ast::{self, FnDefNode, AstNode},
    TextRange, SyntaxNode,
};

use crate::{
    FileId,
    db::SyntaxDatabase,
    descriptors::function::{resolve_local_name, FnId, FnScopes},
    descriptors::module::{
        ModuleId, ModuleTree, ModuleSource, ModuleDescriptor,
        nameres::{ItemMap, InputModuleItems, FileItems}
    },
    input::SourceRootId,
    loc2id::{IdDatabase, DefId, DefLoc},
    syntax_ptr::LocalSyntaxPtr,
    Cancelable,
};

pub(crate) use self::path::{Path, PathKind};
pub(crate) use self::module::nameres::FileItemId;

salsa::query_group! {
    pub(crate) trait DescriptorDatabase: SyntaxDatabase + IdDatabase {
        fn fn_scopes(fn_id: FnId) -> Arc<FnScopes> {
            type FnScopesQuery;
            use fn function::imp::fn_scopes;
        }

        fn _file_items(file_id: FileId) -> Arc<FileItems> {
            type FileItemsQuery;
            storage dependencies;
            use fn module::nameres::file_items;
        }

        fn _file_item(file_id: FileId, file_item_id: FileItemId) -> SyntaxNode {
            type FileItemQuery;
            storage dependencies;
            use fn module::nameres::file_item;
        }

        fn _input_module_items(source_root_id: SourceRootId, module_id: ModuleId) -> Cancelable<Arc<InputModuleItems>> {
            type InputModuleItemsQuery;
            use fn module::nameres::input_module_items;
        }
        fn _item_map(source_root_id: SourceRootId) -> Cancelable<Arc<ItemMap>> {
            type ItemMapQuery;
            use fn module::nameres::item_map;
        }
        fn _module_tree(source_root_id: SourceRootId) -> Cancelable<Arc<ModuleTree>> {
            type ModuleTreeQuery;
            use fn module::imp::module_tree;
        }
        fn _fn_syntax(fn_id: FnId) -> FnDefNode {
            type FnSyntaxQuery;
            // Don't retain syntax trees in memory
            storage dependencies;
            use fn function::imp::fn_syntax;
        }
        fn _submodules(source: ModuleSource) -> Cancelable<Arc<Vec<module::imp::Submodule>>> {
            type SubmodulesQuery;
            use fn module::imp::submodules;
        }
    }
}

pub(crate) enum Def {
    Module(ModuleDescriptor),
    Item,
}

impl DefId {
    pub(crate) fn resolve(self, db: &impl DescriptorDatabase) -> Cancelable<Def> {
        let loc = db.id_maps().def_loc(self);
        let res = match loc {
            DefLoc::Module { id, source_root } => {
                let descr = ModuleDescriptor::new(db, source_root, id)?;
                Def::Module(descr)
            }
            DefLoc::Item { .. } => Def::Item,
        };
        Ok(res)
    }
}

#[derive(Debug)]
pub struct ReferenceDescriptor {
    pub range: TextRange,
    pub name: String,
}

#[derive(Debug)]
pub struct DeclarationDescriptor<'a> {
    pat: ast::BindPat<'a>,
    pub range: TextRange,
}

impl<'a> DeclarationDescriptor<'a> {
    pub fn new(pat: ast::BindPat) -> DeclarationDescriptor {
        let range = pat.syntax().range();

        DeclarationDescriptor { pat, range }
    }

    pub fn find_all_refs(&self) -> Vec<ReferenceDescriptor> {
        let name_ptr = LocalSyntaxPtr::new(self.pat.syntax());

        let fn_def = match self.pat.syntax().ancestors().find_map(ast::FnDef::cast) {
            Some(def) => def,
            None => return Default::default(),
        };

        let fn_scopes = FnScopes::new(fn_def);

        let refs: Vec<_> = fn_def
            .syntax()
            .descendants()
            .filter_map(ast::NameRef::cast)
            .filter(|name_ref| match resolve_local_name(*name_ref, &fn_scopes) {
                None => false,
                Some(entry) => entry.ptr() == name_ptr,
            })
            .map(|name_ref| ReferenceDescriptor {
                name: name_ref.syntax().text().to_string(),
                range: name_ref.syntax().range(),
            })
            .collect();

        refs
    }
}
