//! This crate defines the core datastructure representing IDE state -- `RootDatabase`.
//!
//! It is mainly a `HirDatabase` for semantic analysis, plus a `SymbolsDatabase`, for fuzzy search.

mod apply_change;

pub mod assists;
pub mod label;
pub mod line_index;
pub mod symbol_index;
pub mod defs;
pub mod items_locator;
pub mod source_change;
pub mod ty_filter;
pub mod traits;
pub mod helpers;
pub mod path_transform;

pub mod search;
pub mod rename;
pub mod active_parameter;

use std::{fmt, mem::ManuallyDrop, sync::Arc};

use base_db::{
    salsa::{self, Durability},
    AnchoredPath, CrateId, FileId, FileLoader, FileLoaderDelegate, SourceDatabase, Upcast,
};
use hir::db::{AstDatabase, DefDatabase, HirDatabase};
use rustc_hash::FxHashSet;

use crate::{line_index::LineIndex, symbol_index::SymbolsDatabase};

/// `base_db` is normally also needed in places where `ide_db` is used, so this re-export is for convenience.
pub use base_db;

pub type FxIndexSet<T> = indexmap::IndexSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
pub type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[salsa::database(
    base_db::SourceDatabaseStorage,
    base_db::SourceDatabaseExtStorage,
    LineIndexDatabaseStorage,
    symbol_index::SymbolsDatabaseStorage,
    hir::db::InternDatabaseStorage,
    hir::db::AstDatabaseStorage,
    hir::db::DefDatabaseStorage,
    hir::db::HirDatabaseStorage
)]
pub struct RootDatabase {
    // We use `ManuallyDrop` here because every codegen unit that contains a
    // `&RootDatabase -> &dyn OtherDatabase` cast will instantiate its drop glue in the vtable,
    // which duplicates `Weak::drop` and `Arc::drop` tens of thousands of times, which makes
    // compile times of all `ide_*` and downstream crates suffer greatly.
    storage: ManuallyDrop<salsa::Storage<RootDatabase>>,
}

impl Drop for RootDatabase {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.storage);
        }
    }
}

impl fmt::Debug for RootDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RootDatabase").finish()
    }
}

impl Upcast<dyn AstDatabase> for RootDatabase {
    fn upcast(&self) -> &(dyn AstDatabase + 'static) {
        &*self
    }
}

impl Upcast<dyn DefDatabase> for RootDatabase {
    fn upcast(&self) -> &(dyn DefDatabase + 'static) {
        &*self
    }
}

impl Upcast<dyn HirDatabase> for RootDatabase {
    fn upcast(&self) -> &(dyn HirDatabase + 'static) {
        &*self
    }
}

impl FileLoader for RootDatabase {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        FileLoaderDelegate(self).file_text(file_id)
    }
    fn resolve_path(&self, path: AnchoredPath) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<FxHashSet<CrateId>> {
        FileLoaderDelegate(self).relevant_crates(file_id)
    }
}

impl salsa::Database for RootDatabase {}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        RootDatabase::new(None)
    }
}

impl RootDatabase {
    pub fn new(lru_capacity: Option<usize>) -> RootDatabase {
        let mut db = RootDatabase { storage: ManuallyDrop::new(salsa::Storage::default()) };
        db.set_crate_graph_with_durability(Default::default(), Durability::HIGH);
        db.set_local_roots_with_durability(Default::default(), Durability::HIGH);
        db.set_library_roots_with_durability(Default::default(), Durability::HIGH);
        db.set_enable_proc_attr_macros(Default::default());
        db.update_lru_capacity(lru_capacity);
        db
    }

    pub fn update_lru_capacity(&mut self, lru_capacity: Option<usize>) {
        let lru_capacity = lru_capacity.unwrap_or(base_db::DEFAULT_LRU_CAP);
        base_db::ParseQuery.in_db_mut(self).set_lru_capacity(lru_capacity);
        hir::db::ParseMacroExpansionQuery.in_db_mut(self).set_lru_capacity(lru_capacity);
        hir::db::MacroExpandQuery.in_db_mut(self).set_lru_capacity(lru_capacity);
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase { storage: ManuallyDrop::new(self.storage.snapshot()) })
    }
}

#[salsa::query_group(LineIndexDatabaseStorage)]
pub trait LineIndexDatabase: base_db::SourceDatabase {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &dyn LineIndexDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolKind {
    Attribute,
    BuiltinAttr,
    Const,
    ConstParam,
    Derive,
    Enum,
    Field,
    Function,
    Impl,
    Label,
    LifetimeParam,
    Local,
    Macro,
    Module,
    SelfParam,
    Static,
    Struct,
    ToolModule,
    Trait,
    TypeAlias,
    TypeParam,
    Union,
    ValueParam,
    Variant,
}

impl From<hir::MacroKind> for SymbolKind {
    fn from(it: hir::MacroKind) -> Self {
        match it {
            hir::MacroKind::Declarative | hir::MacroKind::BuiltIn | hir::MacroKind::ProcMacro => {
                SymbolKind::Macro
            }
            hir::MacroKind::Derive => SymbolKind::Derive,
            hir::MacroKind::Attr => SymbolKind::Attribute,
        }
    }
}

#[cfg(test)]
mod tests {
    mod sourcegen_lints;
}
