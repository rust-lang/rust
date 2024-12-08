//! Database used for testing `hir_def`.

use std::{fmt, panic, sync::Mutex};

use base_db::{
    ra_salsa::{self, Durability},
    AnchoredPath, CrateId, FileLoader, FileLoaderDelegate, SourceDatabase, Upcast,
};
use hir_expand::{db::ExpandDatabase, files::FilePosition, InFile};
use span::{EditionedFileId, FileId};
use syntax::{algo, ast, AstNode};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    nameres::{DefMap, ModuleSource},
    src::HasSource,
    LocalModuleId, Lookup, ModuleDefId, ModuleId,
};

#[ra_salsa::database(
    base_db::SourceRootDatabaseStorage,
    base_db::SourceDatabaseStorage,
    hir_expand::db::ExpandDatabaseStorage,
    crate::db::InternDatabaseStorage,
    crate::db::DefDatabaseStorage
)]
pub(crate) struct TestDB {
    storage: ra_salsa::Storage<TestDB>,
    events: Mutex<Option<Vec<ra_salsa::Event>>>,
}

impl Default for TestDB {
    fn default() -> Self {
        let mut this = Self { storage: Default::default(), events: Default::default() };
        this.setup_syntax_context_root();
        this.set_expand_proc_attr_macros_with_durability(true, Durability::HIGH);
        this
    }
}

impl Upcast<dyn ExpandDatabase> for TestDB {
    fn upcast(&self) -> &(dyn ExpandDatabase + 'static) {
        self
    }
}

impl Upcast<dyn DefDatabase> for TestDB {
    fn upcast(&self) -> &(dyn DefDatabase + 'static) {
        self
    }
}

impl ra_salsa::Database for TestDB {
    fn salsa_event(&self, event: ra_salsa::Event) {
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
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<[CrateId]> {
        FileLoaderDelegate(self).relevant_crates(file_id)
    }
}

impl TestDB {
    pub(crate) fn module_for_file(&self, file_id: FileId) -> ModuleId {
        for &krate in self.relevant_crates(file_id).iter() {
            let crate_def_map = self.crate_def_map(krate);
            for (local_id, data) in crate_def_map.modules() {
                if data.origin.file_id().map(EditionedFileId::file_id) == Some(file_id) {
                    return crate_def_map.module_id(local_id);
                }
            }
        }
        panic!("Can't find module for file")
    }

    pub(crate) fn module_at_position(&self, position: FilePosition) -> ModuleId {
        let file_module = self.module_for_file(position.file_id.file_id());
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
                    return def_map.module_id(DefMap::ROOT);
                }
            }
        }
    }

    /// Finds the smallest/innermost module in `def_map` containing `position`.
    fn mod_at_position(&self, def_map: &DefMap, position: FilePosition) -> LocalModuleId {
        let mut size = None;
        let mut res = DefMap::ROOT;
        for (module, data) in def_map.modules() {
            let src = data.definition_source(self);
            if src.file_id != position.file_id {
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
            if file_id != position.file_id {
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
                let expr_id = source_map
                    .node_expr(InFile::new(position.file_id.into(), &expr))?
                    .as_expr()
                    .unwrap();
                let scope = scopes.scope_for(expr_id).unwrap();
                Some(scope)
            });

        for scope in scope_iter {
            let mut containing_blocks =
                scopes.scope_chain(Some(scope)).filter_map(|scope| scopes.block(scope));

            if let Some(block) = containing_blocks.next().map(|block| self.block_def_map(block)) {
                return Some(block);
            }
        }

        None
    }

    pub(crate) fn log(&self, f: impl FnOnce()) -> Vec<ra_salsa::Event> {
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
                ra_salsa::EventKind::WillExecute { database_key } => {
                    Some(format!("{:?}", database_key.debug(self)))
                }
                _ => None,
            })
            .collect()
    }
}
