use std::sync::Arc;

use ra_db::{
    BaseDatabase, FileId, Canceled,
    salsa::{self, Database},
};

use crate::{symbol_index, LineIndex};

#[salsa::database(
    ra_db::FilesDatabase,
    ra_db::SyntaxDatabase,
    LineIndexDatabase,
    symbol_index::SymbolsDatabase,
    hir::db::HirDatabase
)]
#[derive(Debug)]
pub(crate) struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
    interner: Arc<hir::HirInterner>,
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
        };
        db.query_mut(ra_db::CrateGraphQuery)
            .set((), Default::default());
        db.query_mut(ra_db::LocalRootsQuery)
            .set((), Default::default());
        db.query_mut(ra_db::LibraryRootsQuery)
            .set((), Default::default());
        db
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            runtime: self.runtime.snapshot(self),
            interner: Arc::clone(&self.interner),
        })
    }
}

impl BaseDatabase for RootDatabase {}

impl AsRef<hir::HirInterner> for RootDatabase {
    fn as_ref(&self) -> &hir::HirInterner {
        &self.interner
    }
}

#[salsa::query_group]
pub(crate) trait LineIndexDatabase: ra_db::FilesDatabase + BaseDatabase {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &impl ra_db::FilesDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}
