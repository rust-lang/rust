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
mod function;
mod module;
mod path;
mod arena;

use std::ops::Index;

use ra_syntax::{SyntaxNodeRef, SyntaxNode};
use ra_db::{LocationIntener, SourceRootId, FileId, Cancelable};

use crate::{
    db::HirDatabase,
    arena::{Arena, Id},
};

pub use self::{
    path::{Path, PathKind},
    module::{Module, ModuleId, Problem, nameres::ItemMap},
    function::{Function, FnScopes},
};

pub use self::function::FnSignatureInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FnId(u32);
ra_db::impl_numeric_id!(FnId);

impl FnId {
    pub fn from_loc(
        db: &impl AsRef<LocationIntener<SourceItemId, FnId>>,
        loc: &SourceItemId,
    ) -> FnId {
        db.as_ref().loc2id(loc)
    }
    pub fn loc(self, db: &impl AsRef<LocationIntener<SourceItemId, FnId>>) -> SourceItemId {
        db.as_ref().id2loc(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(u32);
ra_db::impl_numeric_id!(DefId);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DefLoc {
    Module {
        id: ModuleId,
        source_root: SourceRootId,
    },
    Item {
        source_item_id: SourceItemId,
    },
}

impl DefId {
    pub fn loc(self, db: &impl AsRef<LocationIntener<DefLoc, DefId>>) -> DefLoc {
        db.as_ref().id2loc(self)
    }
}

impl DefLoc {
    pub fn id(&self, db: &impl AsRef<LocationIntener<DefLoc, DefId>>) -> DefId {
        db.as_ref().loc2id(&self)
    }
}

pub enum Def {
    Module(Module),
    Item,
}

impl DefId {
    pub fn resolve(self, db: &impl HirDatabase) -> Cancelable<Def> {
        let loc = self.loc(db);
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
pub(crate) type SourceFileItemId = Id<SyntaxNode>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceItemId {
    file_id: FileId,
    item_id: SourceFileItemId,
}

/// Maps item's `SyntaxNode`s to `SourceFileItemId` and back.
#[derive(Debug, PartialEq, Eq, Default)]
pub struct SourceFileItems {
    arena: Arena<SyntaxNode>,
}

impl SourceFileItems {
    fn alloc(&mut self, item: SyntaxNode) -> SourceFileItemId {
        self.arena.alloc(item)
    }
    pub fn id_of(&self, item: SyntaxNodeRef) -> SourceFileItemId {
        let (id, _item) = self
            .arena
            .iter()
            .find(|(_id, i)| i.borrowed() == item)
            .unwrap();
        id
    }
}

impl Index<SourceFileItemId> for SourceFileItems {
    type Output = SyntaxNode;
    fn index(&self, idx: SourceFileItemId) -> &SyntaxNode {
        &self.arena[idx]
    }
}
