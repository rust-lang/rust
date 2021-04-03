//! Database used for testing `hir_def`.

use std::{
    fmt, panic,
    sync::{Arc, Mutex},
};

use base_db::{salsa, CrateId, FileId, FileLoader, FileLoaderDelegate, FilePosition, Upcast};
use base_db::{AnchoredPath, SourceDatabase};
use hir_expand::diagnostics::Diagnostic;
use hir_expand::diagnostics::DiagnosticSinkBuilder;
use hir_expand::{db::AstDatabase, InFile};
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use syntax::{algo, ast, AstNode, TextRange, TextSize};
use test_utils::extract_annotations;

use crate::{db::DefDatabase, nameres::DefMap, src::HasSource, Lookup, ModuleDefId, ModuleId};

#[salsa::database(
    base_db::SourceDatabaseExtStorage,
    base_db::SourceDatabaseStorage,
    hir_expand::db::AstDatabaseStorage,
    crate::db::InternDatabaseStorage,
    crate::db::DefDatabaseStorage
)]
#[derive(Default)]
pub(crate) struct TestDB {
    storage: salsa::Storage<TestDB>,
    events: Mutex<Option<Vec<salsa::Event>>>,
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
    fn salsa_event(&self, event: salsa::Event) {
        let mut events = self.events.lock().unwrap();
        if let Some(events) = &mut *events {
            events.push(event);
        }
    }
}

impl fmt::Debug for TestDB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestDB").finish()
    }
}

impl panic::RefUnwindSafe for TestDB {}

impl FileLoader for TestDB {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        FileLoaderDelegate(self).file_text(file_id)
    }
    fn resolve_path(&self, path: AnchoredPath) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<FxHashSet<CrateId>> {
        FileLoaderDelegate(self).relevant_crates(file_id)
    }
}

impl TestDB {
    pub(crate) fn module_for_file(&self, file_id: FileId) -> ModuleId {
        for &krate in self.relevant_crates(file_id).iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (local_id, data) in crate_def_map.modules() {
                if data.origin.file_id() == Some(file_id) {
                    return crate_def_map.module_id(local_id);
                }
            }
        }
        panic!("Can't find module for file")
    }

    pub(crate) fn module_at_position(&self, position: FilePosition) -> ModuleId {
        let file_module = self.module_for_file(position.file_id);
        let mut def_map = file_module.def_map(self);

        def_map = match self.block_at_position(&def_map, position) {
            Some(it) => it,
            None => return file_module,
        };
        loop {
            let new_map = self.block_at_position(&def_map, position);
            match new_map {
                Some(new_block) if !Arc::ptr_eq(&new_block, &def_map) => {
                    def_map = new_block;
                }
                _ => {
                    // FIXME: handle `mod` inside block expression
                    return def_map.module_id(def_map.root());
                }
            }
        }
    }

    fn block_at_position(&self, def_map: &DefMap, position: FilePosition) -> Option<Arc<DefMap>> {
        // Find the smallest (innermost) function in `def_map` containing the cursor.
        let mut size = None;
        let mut fn_def = None;
        for (_, module) in def_map.modules() {
            let file_id = module.definition_source(self).file_id;
            if file_id != position.file_id.into() {
                continue;
            }
            for decl in module.scope.declarations() {
                if let ModuleDefId::FunctionId(it) = decl {
                    let range = it.lookup(self).source(self).value.syntax().text_range();

                    if !range.contains(position.offset) {
                        continue;
                    }

                    let new_size = match size {
                        None => range.len(),
                        Some(size) => {
                            if range.len() < size {
                                range.len()
                            } else {
                                size
                            }
                        }
                    };
                    if size != Some(new_size) {
                        size = Some(new_size);
                        fn_def = Some(it);
                    }
                }
            }
        }

        // Find the innermost block expression that has a `DefMap`.
        let def_with_body = fn_def?.into();
        let (_, source_map) = self.body_with_source_map(def_with_body);
        let scopes = self.expr_scopes(def_with_body);
        let root = self.parse(position.file_id);

        let scope_iter = algo::ancestors_at_offset(&root.syntax_node(), position.offset)
            .filter_map(|node| {
                let block = ast::BlockExpr::cast(node)?;
                let expr = ast::Expr::from(block);
                let expr_id = source_map.node_expr(InFile::new(position.file_id.into(), &expr))?;
                let scope = scopes.scope_for(expr_id).unwrap();
                Some(scope)
            });

        for scope in scope_iter {
            let containing_blocks =
                scopes.scope_chain(Some(scope)).filter_map(|scope| scopes.block(scope));

            for block in containing_blocks {
                if let Some(def_map) = self.block_def_map(block) {
                    return Some(def_map);
                }
            }
        }

        None
    }

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
                // This pretty horrible, but `Debug` is the only way to inspect
                // QueryDescriptor at the moment.
                salsa::EventKind::WillExecute { database_key } => {
                    Some(format!("{:?}", database_key.debug(self)))
                }
                _ => None,
            })
            .collect()
    }

    pub(crate) fn extract_annotations(&self) -> FxHashMap<FileId, Vec<(TextRange, String)>> {
        let mut files = Vec::new();
        let crate_graph = self.crate_graph();
        for krate in crate_graph.iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (module_id, _) in crate_def_map.modules() {
                let file_id = crate_def_map[module_id].origin.file_id();
                files.extend(file_id)
            }
        }
        assert!(!files.is_empty());
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

    pub(crate) fn diagnostics<F: FnMut(&dyn Diagnostic)>(&self, mut cb: F) {
        let crate_graph = self.crate_graph();
        for krate in crate_graph.iter() {
            let crate_def_map = self.crate_def_map(krate);

            let mut sink = DiagnosticSinkBuilder::new().build(&mut cb);
            for (module_id, module) in crate_def_map.modules() {
                crate_def_map.add_diagnostics(self, module_id, &mut sink);

                for decl in module.scope.declarations() {
                    if let ModuleDefId::FunctionId(it) = decl {
                        let source_map = self.body_with_source_map(it.into()).1;
                        source_map.add_diagnostics(self, &mut sink);
                    }
                }
            }
        }
    }

    pub(crate) fn check_diagnostics(&self) {
        let db: &TestDB = self;
        let annotations = db.extract_annotations();
        assert!(!annotations.is_empty());

        let mut actual: FxHashMap<FileId, Vec<(TextRange, String)>> = FxHashMap::default();
        db.diagnostics(|d| {
            let src = d.display_source();
            let root = db.parse_or_expand(src.file_id).unwrap();

            let node = src.map(|ptr| ptr.to_node(&root));
            let frange = node.as_ref().original_file_range(db);

            let message = d.message();
            actual.entry(frange.file_id).or_default().push((frange.range, message));
        });

        for (file_id, diags) in actual.iter_mut() {
            diags.sort_by_key(|it| it.0.start());
            let text = db.file_text(*file_id);
            // For multiline spans, place them on line start
            for (range, content) in diags {
                if text[*range].contains('\n') {
                    *range = TextRange::new(range.start(), range.start() + TextSize::from(1));
                    *content = format!("... {}", content);
                }
            }
        }

        assert_eq!(annotations, actual);
    }

    pub(crate) fn check_no_diagnostics(&self) {
        let db: &TestDB = self;
        let annotations = db.extract_annotations();
        assert!(annotations.is_empty());

        let mut has_diagnostics = false;
        db.diagnostics(|_| {
            has_diagnostics = true;
        });

        assert!(!has_diagnostics);
    }
}
