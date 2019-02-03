use std::{
    sync::Arc,
    time,
};

use ra_db::{
    CheckCanceled, FileId, Canceled, SourceDatabase,
    salsa,
};

use crate::{LineIndex, symbol_index::{self, SymbolsDatabase}};

#[salsa::database(
    ra_db::SourceDatabaseStorage,
    LineIndexDatabaseStorage,
    symbol_index::SymbolsDatabaseStorage,
    hir::db::HirDatabaseStorage,
    hir::db::PersistentHirDatabaseStorage
)]
#[derive(Debug)]
pub(crate) struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
    interner: Arc<hir::HirInterner>,
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
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        let mut db = RootDatabase {
            runtime: salsa::Runtime::default(),
            interner: Default::default(),
            last_gc: time::Instant::now(),
            last_gc_check: time::Instant::now(),
        };
        db.set_crate_graph(Default::default());
        db.set_local_roots(Default::default());
        db.set_library_roots(Default::default());
        db
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            runtime: self.runtime.snapshot(self),
            interner: Arc::clone(&self.interner),
            last_gc: self.last_gc.clone(),
            last_gc_check: self.last_gc_check.clone(),
        })
    }
}

impl AsRef<hir::HirInterner> for RootDatabase {
    fn as_ref(&self) -> &hir::HirInterner {
        &self.interner
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
