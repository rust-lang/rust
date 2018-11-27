//! HIR (previsouly known as descriptors) provides a high-level OO acess to Rust
//! code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, there relation between syntax and HIR is many-to-one.

pub(crate) mod function;
pub(crate) mod module;
pub(crate) mod db;
mod path;

use ra_syntax::{
    ast::{self, AstNode},
    TextRange,
};

use crate::{
    hir::db::HirDatabase,
    hir::function::{resolve_local_name, FnScopes},
    loc2id::{DefId, DefLoc},
    syntax_ptr::LocalSyntaxPtr,
    Cancelable,
};

pub(crate) use self::{
    path::{Path, PathKind},
    module::{ModuleDescriptor, nameres::FileItemId},
    function::FunctionDescriptor,
};

pub(crate) enum Def {
    Module(ModuleDescriptor),
    Item,
}

impl DefId {
    pub(crate) fn resolve(self, db: &impl HirDatabase) -> Cancelable<Def> {
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
