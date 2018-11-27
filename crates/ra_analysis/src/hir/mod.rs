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

use crate::{
    hir::db::HirDatabase,
    loc2id::{DefId, DefLoc},
    Cancelable,
};

pub(crate) use self::{
    path::{Path, PathKind},
    module::{Module, ModuleId, Problem, nameres::FileItemId},
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
