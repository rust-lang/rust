//! HIR (previsouly known as descriptors) provides a high-level OO acess to Rust
//! code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, there relation between syntax and HIR is many-to-one.

pub(crate) mod db;
mod query_definitions;
mod function;
mod module;
mod path;

use std::ops::Index;

use ra_syntax::{SyntaxNodeRef, SyntaxNode};

use crate::{
    hir::db::HirDatabase,
    loc2id::{DefId, DefLoc},
    Cancelable,
    arena::{Arena, Id},
};

pub(crate) use self::{
    path::{Path, PathKind},
    module::{Module, ModuleId, Problem},
    function::{Function, FnScopes},
};

pub use self::function::FnSignatureInfo;

pub(crate) enum Def {
    Module(Module),
    Item,
}

impl DefId {
    pub(crate) fn resolve(self, db: &impl HirDatabase) -> Cancelable<Def> {
        let loc = db.id_maps().def_loc(self);
        let res = match loc {
            DefLoc::Module { id, source_root } => {
                let descr = Module::new(db, source_root, id)?;
                Def::Module(descr)
            }
            DefLoc::Item { .. } => Def::Item,
        };
        Ok(res)
    }
}

/// Identifier of item within a specific file. This is stable over reparses, so
/// it's OK to use it as a salsa key/value.
pub(crate) type FileItemId = Id<SyntaxNode>;

/// Maps item's `SyntaxNode`s to `FileItemId` and back.
#[derive(Debug, PartialEq, Eq, Default)]
pub(crate) struct FileItems {
    arena: Arena<SyntaxNode>,
}

impl FileItems {
    fn alloc(&mut self, item: SyntaxNode) -> FileItemId {
        self.arena.alloc(item)
    }
    fn id_of(&self, item: SyntaxNodeRef) -> FileItemId {
        let (id, _item) = self
            .arena
            .iter()
            .find(|(_id, i)| i.borrowed() == item)
            .unwrap();
        id
    }
}

impl Index<FileItemId> for FileItems {
    type Output = SyntaxNode;
    fn index(&self, idx: FileItemId) -> &SyntaxNode {
        &self.arena[idx]
    }
}
