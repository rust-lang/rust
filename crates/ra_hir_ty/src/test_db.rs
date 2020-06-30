//! Database used for testing `hir`.

use std::{
    panic,
    sync::{Arc, Mutex},
};

use hir_def::{db::DefDatabase, AssocItemId, ModuleDefId, ModuleId};
use hir_expand::{db::AstDatabase, diagnostics::DiagnosticSink};
use ra_db::{salsa, CrateId, FileId, FileLoader, FileLoaderDelegate, SourceDatabase, Upcast};
use ra_syntax::TextRange;
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::format_to;
use test_utils::extract_annotations;

use crate::{
    db::HirDatabase, diagnostics::Diagnostic, expr::ExprValidator,
    unsafe_validation::UnsafeValidator,
};

#[salsa::database(
    ra_db::SourceDatabaseExtStorage,
    ra_db::SourceDatabaseStorage,
    hir_expand::db::AstDatabaseStorage,
    hir_def::db::InternDatabaseStorage,
    hir_def::db::DefDatabaseStorage,
    crate::db::HirDatabaseStorage
)]
#[derive(Debug, Default)]
pub struct TestDB {
    events: Mutex<Option<Vec<salsa::Event<TestDB>>>>,
    runtime: salsa::Runtime<TestDB>,
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
    fn salsa_runtime(&self) -> &salsa::Runtime<TestDB> {
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

impl salsa::ParallelDatabase for TestDB {
    fn snapshot(&self) -> salsa::Snapshot<TestDB> {
        salsa::Snapshot::new(TestDB {
            events: Default::default(),
            runtime: self.runtime.snapshot(self),
        })
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
    pub fn module_for_file(&self, file_id: FileId) -> ModuleId {
        for &krate in self.relevant_crates(file_id).iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (local_id, data) in crate_def_map.modules.iter() {
                if data.origin.file_id() == Some(file_id) {
                    return ModuleId { krate, local_id };
                }
            }
        }
        panic!("Can't find module for file")
    }

    fn diag<F: FnMut(&dyn Diagnostic)>(&self, mut cb: F) {
        let crate_graph = self.crate_graph();
        for krate in crate_graph.iter() {
            let crate_def_map = self.crate_def_map(krate);

            let mut fns = Vec::new();
            for (module_id, _) in crate_def_map.modules.iter() {
                for decl in crate_def_map[module_id].scope.declarations() {
                    if let ModuleDefId::FunctionId(f) = decl {
                        fns.push(f)
                    }
                }

                for impl_id in crate_def_map[module_id].scope.impls() {
                    let impl_data = self.impl_data(impl_id);
                    for item in impl_data.items.iter() {
                        if let AssocItemId::FunctionId(f) = item {
                            fns.push(*f)
                        }
                    }
                }
            }

            for f in fns {
                let infer = self.infer(f.into());
                let mut sink = DiagnosticSink::new(&mut cb);
                infer.add_diagnostics(self, f, &mut sink);
                let mut validator = ExprValidator::new(f, infer.clone(), &mut sink);
                validator.validate_body(self);
                let mut validator = UnsafeValidator::new(f, infer, &mut sink);
                validator.validate_body(self);
            }
        }
    }

    pub fn diagnostics(&self) -> (String, u32) {
        let mut buf = String::new();
        let mut count = 0;
        self.diag(|d| {
            format_to!(buf, "{:?}: {}\n", d.syntax_node(self).text(), d.message());
            count += 1;
        });
        (buf, count)
    }

    /// Like `diagnostics`, but filtered for a single diagnostic.
    pub fn diagnostic<D: Diagnostic>(&self) -> (String, u32) {
        let mut buf = String::new();
        let mut count = 0;
        self.diag(|d| {
            // We want to filter diagnostics by the particular one we are testing for, to
            // avoid surprising results in tests.
            if d.downcast_ref::<D>().is_some() {
                format_to!(buf, "{:?}: {}\n", d.syntax_node(self).text(), d.message());
                count += 1;
            };
        });
        (buf, count)
    }

    pub fn extract_annotations(&self) -> FxHashMap<FileId, Vec<(TextRange, String)>> {
        let mut files = Vec::new();
        let crate_graph = self.crate_graph();
        for krate in crate_graph.iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (module_id, _) in crate_def_map.modules.iter() {
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
