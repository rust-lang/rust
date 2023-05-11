//! Database used for testing `hir`.

use std::{
    fmt, panic,
    sync::{Arc, Mutex},
};

use base_db::{
    salsa, AnchoredPath, CrateId, FileId, FileLoader, FileLoaderDelegate, SourceDatabase, Upcast,
};
use hir_def::{db::DefDatabase, ModuleId};
use hir_expand::db::ExpandDatabase;
use stdx::hash::{NoHashHashMap, NoHashHashSet};
use syntax::TextRange;
use test_utils::extract_annotations;

#[salsa::database(
    base_db::SourceDatabaseExtStorage,
    base_db::SourceDatabaseStorage,
    hir_expand::db::ExpandDatabaseStorage,
    hir_def::db::InternDatabaseStorage,
    hir_def::db::DefDatabaseStorage,
    crate::db::HirDatabaseStorage
)]
pub(crate) struct TestDB {
    storage: salsa::Storage<TestDB>,
    events: Mutex<Option<Vec<salsa::Event>>>,
}

impl Default for TestDB {
    fn default() -> Self {
        let mut this = Self { storage: Default::default(), events: Default::default() };
        this.set_enable_proc_attr_macros(true);
        this
    }
}

impl fmt::Debug for TestDB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestDB").finish()
    }
}

impl Upcast<dyn ExpandDatabase> for TestDB {
    fn upcast(&self) -> &(dyn ExpandDatabase + 'static) {
        &*self
    }
}

impl Upcast<dyn DefDatabase> for TestDB {
    fn upcast(&self) -> &(dyn DefDatabase + 'static) {
        &*self
    }
}

impl salsa::Database for TestDB {
    fn salsa_event(&self, event: salsa::Event) {
        let mut events = self.events.lock().unwrap();
        if let Some(events) = &mut *events {
            events.push(event);
        }
    }
}

impl salsa::ParallelDatabase for TestDB {
    fn snapshot(&self) -> salsa::Snapshot<TestDB> {
        salsa::Snapshot::new(TestDB {
            storage: self.storage.snapshot(),
            events: Default::default(),
        })
    }
}

impl panic::RefUnwindSafe for TestDB {}

impl FileLoader for TestDB {
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

impl TestDB {
    pub(crate) fn module_for_file_opt(&self, file_id: FileId) -> Option<ModuleId> {
        for &krate in self.relevant_crates(file_id).iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (local_id, data) in crate_def_map.modules() {
                if data.origin.file_id() == Some(file_id) {
                    return Some(crate_def_map.module_id(local_id));
                }
            }
        }
        None
    }

    pub(crate) fn module_for_file(&self, file_id: FileId) -> ModuleId {
        self.module_for_file_opt(file_id).unwrap()
    }

    pub(crate) fn extract_annotations(&self) -> NoHashHashMap<FileId, Vec<(TextRange, String)>> {
        let mut files = Vec::new();
        let crate_graph = self.crate_graph();
        for krate in crate_graph.iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (module_id, _) in crate_def_map.modules() {
                let file_id = crate_def_map[module_id].origin.file_id();
                files.extend(file_id)
            }
        }
        files
            .into_iter()
            .filter_map(|file_id| {
                let text = self.file_text(file_id);
                let annotations = extract_annotations(&text);
                if annotations.is_empty() {
                    return None;
                }
                Some((file_id, annotations))
            })
            .collect()
    }
}

impl TestDB {
    pub(crate) fn log(&self, f: impl FnOnce()) -> Vec<salsa::Event> {
        *self.events.lock().unwrap() = Some(Vec::new());
        f();
        self.events.lock().unwrap().take().unwrap()
    }

    pub(crate) fn log_executed(&self, f: impl FnOnce()) -> Vec<String> {
        let events = self.log(f);
        events
            .into_iter()
            .filter_map(|e| match e.kind {
                // This is pretty horrible, but `Debug` is the only way to inspect
                // QueryDescriptor at the moment.
                salsa::EventKind::WillExecute { database_key } => {
                    Some(format!("{:?}", database_key.debug(self)))
                }
                _ => None,
            })
            .collect()
    }
}
