//! Database used for testing `hir_def`.

use std::{
    panic,
    sync::{Arc, Mutex},
};

use hir_expand::db::AstDatabase;
use ra_db::{salsa, CrateId, FileId, FileLoader, FileLoaderDelegate, Upcast};
use rustc_hash::FxHashSet;

use crate::db::DefDatabase;

#[salsa::database(
    ra_db::SourceDatabaseExtStorage,
    ra_db::SourceDatabaseStorage,
    hir_expand::db::AstDatabaseStorage,
    crate::db::InternDatabaseStorage,
    crate::db::DefDatabaseStorage
)]
#[derive(Debug, Default)]
pub struct TestDB {
    runtime: salsa::Runtime<TestDB>,
    events: Mutex<Option<Vec<salsa::Event<TestDB>>>>,
}

impl Upcast<dyn AstDatabase> for TestDB {
    fn upcast(&self) -> &(dyn AstDatabase + 'static) {
        &*self
    }
}

impl Upcast<dyn DefDatabase> for TestDB {
    fn upcast(&self) -> &(dyn DefDatabase + 'static) {
        &*self
    }
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

impl TestDB {
    pub fn module_for_file(&self, file_id: FileId) -> crate::ModuleId {
        for &krate in self.relevant_crates(file_id).iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (local_id, data) in crate_def_map.modules.iter() {
                if data.origin.file_id() == Some(file_id) {
                    return crate::ModuleId { krate, local_id };
                }
            }
        }
        panic!("Can't find module for file")
    }

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
