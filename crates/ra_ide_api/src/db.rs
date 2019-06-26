use std::{sync::Arc, time};

use ra_db::{
    salsa::{self, Database, Durability},
    Canceled, CheckCanceled, FileId, SourceDatabase,
};

use crate::{
    symbol_index::{self, SymbolsDatabase},
    LineIndex,
};

#[salsa::database(
    ra_db::SourceDatabaseStorage,
    LineIndexDatabaseStorage,
    symbol_index::SymbolsDatabaseStorage,
    hir::db::InternDatabaseStorage,
    hir::db::AstDatabaseStorage,
    hir::db::DefDatabaseStorage,
    hir::db::HirDatabaseStorage
)]
#[derive(Debug)]
pub(crate) struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
    pub(crate) last_gc: time::Instant,
    pub(crate) last_gc_check: time::Instant,
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<RootDatabase> {
        &self.runtime
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
        RootDatabase::new(None)
    }
}

impl RootDatabase {
    pub fn new(lru_capacity: Option<usize>) -> RootDatabase {
        let mut db = RootDatabase {
            runtime: salsa::Runtime::default(),
            last_gc: time::Instant::now(),
            last_gc_check: time::Instant::now(),
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
        })
    }
}

#[salsa::query_group(LineIndexDatabaseStorage)]
pub(crate) trait LineIndexDatabase: ra_db::SourceDatabase + CheckCanceled {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &impl ra_db::SourceDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}
