//! This crate defines the core datastructure representing IDE state -- `RootDatabase`.
//!
//! It is mainly a `HirDatabase` for semantic analysis, plus a `SymbolsDatabase`, for fuzzy search.

pub mod marks;
pub mod line_index;
pub mod line_index_utils;
pub mod feature_flags;
pub mod symbol_index;
pub mod change;
pub mod defs;
pub mod imports_locator;
mod wasm_shims;

use std::sync::Arc;

use ra_db::{
    salsa::{self, Database, Durability},
    Canceled, CheckCanceled, CrateId, FileId, FileLoader, FileLoaderDelegate, RelativePath,
    SourceDatabase, SourceRootId,
};
use rustc_hash::FxHashMap;

use crate::{feature_flags::FeatureFlags, line_index::LineIndex, symbol_index::SymbolsDatabase};

#[salsa::database(
    ra_db::SourceDatabaseStorage,
    ra_db::SourceDatabaseExtStorage,
    LineIndexDatabaseStorage,
    symbol_index::SymbolsDatabaseStorage,
    hir::db::InternDatabaseStorage,
    hir::db::AstDatabaseStorage,
    hir::db::DefDatabaseStorage,
    hir::db::HirDatabaseStorage
)]
#[derive(Debug)]
pub struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
    pub feature_flags: Arc<FeatureFlags>,
    pub(crate) debug_data: Arc<DebugData>,
    pub last_gc: crate::wasm_shims::Instant,
    pub last_gc_check: crate::wasm_shims::Instant,
}

impl FileLoader for RootDatabase {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        FileLoaderDelegate(self).file_text(file_id)
    }
    fn resolve_relative_path(
        &self,
        anchor: FileId,
        relative_path: &RelativePath,
    ) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_relative_path(anchor, relative_path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<Vec<CrateId>> {
        FileLoaderDelegate(self).relevant_crates(file_id)
    }
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<RootDatabase> {
        &self.runtime
    }
    fn salsa_runtime_mut(&mut self) -> &mut salsa::Runtime<Self> {
        &mut self.runtime
    }
    fn on_propagated_panic(&self) -> ! {
        Canceled::throw()
    }
    fn salsa_event(&self, event: impl Fn() -> salsa::Event<RootDatabase>) {
        match event().kind {
            salsa::EventKind::DidValidateMemoizedValue { .. }
            | salsa::EventKind::WillExecute { .. } => {
                self.check_canceled();
            }
            _ => (),
        }
    }
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        RootDatabase::new(None, FeatureFlags::default())
    }
}

impl RootDatabase {
    pub fn new(lru_capacity: Option<usize>, feature_flags: FeatureFlags) -> RootDatabase {
        let mut db = RootDatabase {
            runtime: salsa::Runtime::default(),
            last_gc: crate::wasm_shims::Instant::now(),
            last_gc_check: crate::wasm_shims::Instant::now(),
            feature_flags: Arc::new(feature_flags),
            debug_data: Default::default(),
        };
        db.set_crate_graph_with_durability(Default::default(), Durability::HIGH);
        db.set_local_roots_with_durability(Default::default(), Durability::HIGH);
        db.set_library_roots_with_durability(Default::default(), Durability::HIGH);
        let lru_capacity = lru_capacity.unwrap_or(ra_db::DEFAULT_LRU_CAP);
        db.query_mut(ra_db::ParseQuery).set_lru_capacity(lru_capacity);
        db.query_mut(hir::db::ParseMacroQuery).set_lru_capacity(lru_capacity);
        db.query_mut(hir::db::MacroExpandQuery).set_lru_capacity(lru_capacity);
        db
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            runtime: self.runtime.snapshot(self),
            last_gc: self.last_gc,
            last_gc_check: self.last_gc_check,
            feature_flags: Arc::clone(&self.feature_flags),
            debug_data: Arc::clone(&self.debug_data),
        })
    }
}

#[salsa::query_group(LineIndexDatabaseStorage)]
pub trait LineIndexDatabase: ra_db::SourceDatabase + CheckCanceled {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &impl LineIndexDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}

#[derive(Debug, Default, Clone)]
pub(crate) struct DebugData {
    pub(crate) root_paths: FxHashMap<SourceRootId, String>,
    pub(crate) crate_names: FxHashMap<CrateId, String>,
}

impl DebugData {
    pub(crate) fn merge(&mut self, other: DebugData) {
        self.root_paths.extend(other.root_paths.into_iter());
        self.crate_names.extend(other.crate_names.into_iter());
    }
}
