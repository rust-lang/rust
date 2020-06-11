//! Database used for testing `hir_expand`.

use std::{
    panic,
    sync::{Arc, Mutex},
};

use ra_db::{salsa, CrateId, FileId, FileLoader, FileLoaderDelegate};
use rustc_hash::FxHashSet;

#[salsa::database(
    ra_db::SourceDatabaseExtStorage,
    ra_db::SourceDatabaseStorage,
    crate::db::AstDatabaseStorage
)]
#[derive(Debug, Default)]
pub struct TestDB {
    runtime: salsa::Runtime<TestDB>,
    events: Mutex<Option<Vec<salsa::Event<TestDB>>>>,
}

impl salsa::Database for TestDB {
    fn salsa_runtime(&self) -> &salsa::Runtime<Self> {
        &self.runtime
    }

    fn salsa_runtime_mut(&mut self) -> &mut salsa::Runtime<Self> {
        &mut self.runtime
    }

    fn salsa_event(&self, event: impl Fn() -> salsa::Event<TestDB>) {
        let mut events = self.events.lock().unwrap();
        if let Some(events) = &mut *events {
            events.push(event());
        }
    }
}

impl panic::RefUnwindSafe for TestDB {}

impl FileLoader for TestDB {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        FileLoaderDelegate(self).file_text(file_id)
    }
    fn resolve_path(&self, anchor: FileId, path: &str) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(anchor, path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<FxHashSet<CrateId>> {
        FileLoaderDelegate(self).relevant_crates(file_id)
    }
}
