//! This crate defines the core datastructure representing IDE state -- `RootDatabase`.
//!
//! It is mainly a `HirDatabase` for semantic analysis, plus a `SymbolsDatabase`, for fuzzy search.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod apply_change;

pub mod active_parameter;
pub mod assists;
pub mod defs;
pub mod famous_defs;
pub mod helpers;
pub mod items_locator;
pub mod label;
pub mod line_index;
pub mod path_transform;
pub mod rename;
pub mod rust_doc;
pub mod search;
pub mod source_change;
pub mod symbol_index;
pub mod traits;
pub mod ty_filter;
pub mod use_trivial_constructor;

pub mod imports {
    pub mod import_assets;
    pub mod insert_use;
    pub mod merge_imports;
}

pub mod generated {
    pub mod lints;
}

pub mod syntax_helpers {
    pub mod node_ext;
    pub mod insert_whitespace_into_node;
    pub mod format_string;
    pub mod format_string_exprs;

    pub use parser::LexedStr;
}

use std::{fmt, mem::ManuallyDrop, sync::Arc};

use base_db::{
    salsa::{self, Durability},
    AnchoredPath, CrateId, FileId, FileLoader, FileLoaderDelegate, SourceDatabase, Upcast,
};
use hir::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    symbols::FileSymbolKind,
};
use stdx::hash::NoHashHashSet;

use crate::{line_index::LineIndex, symbol_index::SymbolsDatabase};
pub use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

/// `base_db` is normally also needed in places where `ide_db` is used, so this re-export is for convenience.
pub use base_db;

pub type FxIndexSet<T> = indexmap::IndexSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
pub type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[salsa::database(
    base_db::SourceDatabaseExtStorage,
    base_db::SourceDatabaseStorage,
    hir::db::AstDatabaseStorage,
    hir::db::DefDatabaseStorage,
    hir::db::HirDatabaseStorage,
    hir::db::InternDatabaseStorage,
    LineIndexDatabaseStorage,
    symbol_index::SymbolsDatabaseStorage
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
        unsafe { ManuallyDrop::drop(&mut self.storage) };
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
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<NoHashHashSet<CrateId>> {
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
        db.set_enable_proc_attr_macros(false);
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
    Arc::new(LineIndex::new(&text))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolKind {
    Attribute,
    BuiltinAttr,
    Const,
    ConstParam,
    Derive,
    DeriveHelper,
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
    SelfType,
    Static,
    Struct,
    ToolModule,
    Trait,
    TraitAlias,
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

impl From<FileSymbolKind> for SymbolKind {
    fn from(it: FileSymbolKind) -> Self {
        match it {
            FileSymbolKind::Const => SymbolKind::Const,
            FileSymbolKind::Enum => SymbolKind::Enum,
            FileSymbolKind::Function => SymbolKind::Function,
            FileSymbolKind::Macro => SymbolKind::Macro,
            FileSymbolKind::Module => SymbolKind::Module,
            FileSymbolKind::Static => SymbolKind::Static,
            FileSymbolKind::Struct => SymbolKind::Struct,
            FileSymbolKind::Trait => SymbolKind::Trait,
            FileSymbolKind::TraitAlias => SymbolKind::TraitAlias,
            FileSymbolKind::TypeAlias => SymbolKind::TypeAlias,
            FileSymbolKind::Union => SymbolKind::Union,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl SnippetCap {
    pub const fn new(allow_snippets: bool) -> Option<SnippetCap> {
        if allow_snippets {
            Some(SnippetCap { _private: () })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    mod sourcegen_lints;
}
