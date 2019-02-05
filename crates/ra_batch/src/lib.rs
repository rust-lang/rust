use std::sync::Arc;

use ra_db::{
    FilePosition, FileId, CrateGraph, SourceRoot, SourceRootId, SourceDatabase, salsa,
};
use ra_hir::{db, HirInterner};

#[salsa::database(
    ra_db::SourceDatabaseStorage,
    db::HirDatabaseStorage,
    db::PersistentHirDatabaseStorage
)]
#[derive(Debug)]
pub(crate) struct BatchDatabase {
    runtime: salsa::Runtime<BatchDatabase>,
    interner: Arc<HirInterner>,
    file_counter: u32,
}

impl salsa::Database for BatchDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<BatchDatabase> {
        &self.runtime
    }
}

impl AsRef<HirInterner> for BatchDatabase {
    fn as_ref(&self) -> &HirInterner {
        &self.interner
    }
}
