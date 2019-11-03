use std::{panic, sync::Arc};

use ra_db::{salsa, CrateId, FileId, FileLoader, FileLoaderDelegate};
use relative_path::RelativePath;

#[salsa::database(
    ra_db::SourceDatabaseExtStorage,
    ra_db::SourceDatabaseStorage,
    hir_expand::db::AstDatabaseStorage,
    crate::db::InternDatabaseStorage,
    crate::db::DefDatabase2Storage
)]
#[derive(Debug, Default)]
pub struct TestDB {
    runtime: salsa::Runtime<TestDB>,
}

impl salsa::Database for TestDB {
    fn salsa_runtime(&self) -> &salsa::Runtime<Self> {
        &self.runtime
    }
}

impl panic::RefUnwindSafe for TestDB {}

impl FileLoader for TestDB {
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
