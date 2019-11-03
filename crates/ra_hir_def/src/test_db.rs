//! Database used for testing `hir_def`.

use std::{
    panic,
    sync::{Arc, Mutex},
};

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
    events: Mutex<Option<Vec<salsa::Event<TestDB>>>>,
}

impl salsa::Database for TestDB {
    fn salsa_runtime(&self) -> &salsa::Runtime<Self> {
        &self.runtime
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

impl TestDB {
    pub fn log(&self, f: impl FnOnce()) -> Vec<salsa::Event<TestDB>> {
        *self.events.lock().unwrap() = Some(Vec::new());
        f();
        self.events.lock().unwrap().take().unwrap()
    }

    pub fn log_executed(&self, f: impl FnOnce()) -> Vec<String> {
        let events = self.log(f);
        events
            .into_iter()
            .filter_map(|e| match e.kind {
                // This pretty horrible, but `Debug` is the only way to inspect
                // QueryDescriptor at the moment.
                salsa::EventKind::WillExecute { database_key } => {
                    Some(format!("{:?}", database_key))
                }
                _ => None,
            })
            .collect()
    }
}
