//! Database used for testing `hir_def`.

use std::{
    fmt, panic,
    sync::{Arc, Mutex},
};

use base_db::{
    salsa, AnchoredPath, CrateId, FileId, FileLoader, FileLoaderDelegate, FilePosition,
    SourceDatabase, Upcast,
};
use hir_expand::{db::ExpandDatabase, InFile};
use stdx::hash::NoHashHashSet;
use syntax::{algo, ast, AstNode};

use crate::{
    db::DefDatabase,
    nameres::{DefMap, ModuleSource},
    src::HasSource,
    LocalModuleId, Lookup, ModuleDefId, ModuleId,
};

#[salsa::database(
    base_db::SourceDatabaseExtStorage,
    base_db::SourceDatabaseStorage,
    hir_expand::db::ExpandDatabaseStorage,
    crate::db::InternDatabaseStorage,
    crate::db::DefDatabaseStorage
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
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<NoHashHashSet<CrateId>> {
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
        let module = self.mod_at_position(&def_map, position);

        def_map = match self.block_at_position(&def_map, position) {
            Some(it) => it,
            None => return def_map.module_id(module),
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

    /// Finds the smallest/innermost module in `def_map` containing `position`.
    fn mod_at_position(&self, def_map: &DefMap, position: FilePosition) -> LocalModuleId {
        let mut size = None;
        let mut res = def_map.root();
        for (module, data) in def_map.modules() {
            let src = data.definition_source(self);
            if src.file_id != position.file_id.into() {
                continue;
            }

            let range = match src.value {
                ModuleSource::SourceFile(it) => it.syntax().text_range(),
                ModuleSource::Module(it) => it.syntax().text_range(),
                ModuleSource::BlockExpr(it) => it.syntax().text_range(),
            };

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
                cov_mark::hit!(submodule_in_testdb);
                size = Some(new_size);
                res = module;
            }
        }

        res
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
