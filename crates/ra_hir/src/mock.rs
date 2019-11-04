//! FIXME: write short doc here

use std::{panic, sync::Arc};

use hir_def::{db::DefDatabase2, ModuleId};
use hir_expand::diagnostics::DiagnosticSink;
use parking_lot::Mutex;
use ra_db::{
    salsa, CrateId, FileId, FileLoader, FileLoaderDelegate, RelativePath, SourceDatabase,
    SourceRootId,
};

use crate::{db, debug::HirDebugHelper};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

#[salsa::database(
    ra_db::SourceDatabaseExtStorage,
    ra_db::SourceDatabaseStorage,
    db::InternDatabaseStorage,
    db::AstDatabaseStorage,
    db::DefDatabaseStorage,
    db::DefDatabase2Storage,
    db::HirDatabaseStorage
)]
#[derive(Debug)]
pub struct MockDatabase {
    events: Mutex<Option<Vec<salsa::Event<MockDatabase>>>>,
    runtime: salsa::Runtime<MockDatabase>,
}

impl panic::RefUnwindSafe for MockDatabase {}

impl FileLoader for MockDatabase {
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

// FIXME: improve `WithFixture` to bring useful hir debugging back
impl HirDebugHelper for MockDatabase {
    fn crate_name(&self, _krate: CrateId) -> Option<String> {
        None
    }

    fn file_path(&self, _file_id: FileId) -> Option<String> {
        None
    }
}

impl MockDatabase {
    pub fn diagnostics(&self) -> String {
        let mut buf = String::new();
        let crate_graph = self.crate_graph();
        for krate in crate_graph.iter().next() {
            let crate_def_map = self.crate_def_map(krate);
            for (module_id, _) in crate_def_map.modules.iter() {
                let module_id = ModuleId { krate, module_id };
                let module = crate::Module::from(module_id);
                module.diagnostics(
                    self,
                    &mut DiagnosticSink::new(|d| {
                        buf += &format!("{:?}: {}\n", d.syntax_node(self).text(), d.message());
                    }),
                )
            }
        }
        buf
    }
}

impl salsa::Database for MockDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<MockDatabase> {
        &self.runtime
    }

    fn salsa_event(&self, event: impl Fn() -> salsa::Event<MockDatabase>) {
        let mut events = self.events.lock();
        if let Some(events) = &mut *events {
            events.push(event());
        }
    }
}

impl Default for MockDatabase {
    fn default() -> MockDatabase {
        let mut db =
            MockDatabase { events: Default::default(), runtime: salsa::Runtime::default() };
        db.set_crate_graph(Default::default());
        db
    }
}

impl salsa::ParallelDatabase for MockDatabase {
    fn snapshot(&self) -> salsa::Snapshot<MockDatabase> {
        salsa::Snapshot::new(MockDatabase {
            events: Default::default(),
            runtime: self.runtime.snapshot(self),
        })
    }
}

impl MockDatabase {
    pub fn log(&self, f: impl FnOnce()) -> Vec<salsa::Event<MockDatabase>> {
        *self.events.lock() = Some(Vec::new());
        f();
        self.events.lock().take().unwrap()
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
