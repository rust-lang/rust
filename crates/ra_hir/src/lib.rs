//! HIR (previsouly known as descriptors) provides a high-level OO acess to Rust
//! code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, there relation between syntax and HIR is many-to-one.

macro_rules! ctry {
    ($expr:expr) => {
        match $expr {
            None => return Ok(None),
            Some(it) => it,
        }
    };
}

pub mod db;
#[cfg(test)]
mod mock;
mod query_definitions;
mod path;
mod arena;
pub mod source_binder;

mod krate;
mod module;
mod function;

use std::ops::Index;

use ra_syntax::{SyntaxNodeRef, SyntaxNode};
use ra_db::{LocationIntener, SourceRootId, FileId, Cancelable};

use crate::{
    db::HirDatabase,
    arena::{Arena, Id},
};

pub use self::{
    path::{Path, PathKind},
    krate::Crate,
    module::{Module, ModuleId, Problem, nameres::ItemMap},
    function::{Function, FnScopes},
};

pub use self::function::FnSignatureInfo;

/// Def's are a core concept of hir. A `Def` is an Item (function, module, etc)
/// in a specific module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(u32);
ra_db::impl_numeric_id!(DefId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DefKind {
    Module,
    Function,
    Item,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DefLoc {
    pub(crate) kind: DefKind,
    source_root_id: SourceRootId,
    module_id: ModuleId,
    source_item_id: SourceItemId,
}

impl DefId {
    pub(crate) fn loc(self, db: &impl AsRef<LocationIntener<DefLoc, DefId>>) -> DefLoc {
        db.as_ref().id2loc(self)
    }
}

impl DefLoc {
    pub(crate) fn id(&self, db: &impl AsRef<LocationIntener<DefLoc, DefId>>) -> DefId {
        db.as_ref().loc2id(&self)
    }
}

pub enum Def {
    Module(Module),
    Function(Function),
    Item,
}

impl DefId {
    pub fn resolve(self, db: &impl HirDatabase) -> Cancelable<Def> {
        let loc = self.loc(db);
        let res = match loc.kind {
            DefKind::Module => {
                let module = Module::new(db, loc.source_root_id, loc.module_id)?;
                Def::Module(module)
            }
            DefKind::Function => {
                let function = Function::new(self);
                Def::Function(function)
            }
            DefKind::Item => Def::Item,
        };
        Ok(res)
    }
}

/// Identifier of item within a specific file. This is stable over reparses, so
/// it's OK to use it as a salsa key/value.
pub(crate) type SourceFileItemId = Id<SyntaxNode>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceItemId {
    file_id: FileId,
    item_id: SourceFileItemId,
}

/// Maps item's `SyntaxNode`s to `SourceFileItemId` and back.
#[derive(Debug, PartialEq, Eq)]
pub struct SourceFileItems {
    file_id: FileId,
    arena: Arena<SyntaxNode>,
}

impl SourceFileItems {
    fn new(file_id: FileId) -> SourceFileItems {
        SourceFileItems {
            file_id,
            arena: Arena::default(),
        }
    }

    fn alloc(&mut self, item: SyntaxNode) -> SourceFileItemId {
        self.arena.alloc(item)
    }
    pub fn id_of(&self, file_id: FileId, item: SyntaxNodeRef) -> SourceFileItemId {
        assert_eq!(
            self.file_id, file_id,
            "SourceFileItems: wrong file, expected {:?}, got {:?}",
            self.file_id, file_id
        );
        self.id_of_unchecked(item)
    }
    fn id_of_unchecked(&self, item: SyntaxNodeRef) -> SourceFileItemId {
        let (id, _item) = self
            .arena
            .iter()
            .find(|(_id, i)| i.borrowed() == item)
            .unwrap();
        id
    }
    pub fn id_of_source_file(&self) -> SourceFileItemId {
        let (id, _syntax) = self.arena.iter().next().unwrap();
        id
    }
}

impl Index<SourceFileItemId> for SourceFileItems {
    type Output = SyntaxNode;
    fn index(&self, idx: SourceFileItemId) -> &SyntaxNode {
        &self.arena[idx]
    }
}
