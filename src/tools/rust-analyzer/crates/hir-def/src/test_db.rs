//! Database used for testing `hir_def`.

use std::{fmt, panic, sync::Mutex};

use base_db::{
    Crate, CrateGraphBuilder, CratesMap, FileSourceRootInput, FileText, RootQueryDb,
    SourceDatabase, SourceRoot, SourceRootId, SourceRootInput,
};
use hir_expand::{InFile, files::FilePosition};
use salsa::{AsDynDatabase, Durability};
use span::FileId;
use syntax::{AstNode, algo, ast};
use triomphe::Arc;

use crate::{
    LocalModuleId, Lookup, ModuleDefId, ModuleId,
    db::DefDatabase,
    nameres::{DefMap, ModuleSource, block_def_map, crate_def_map},
    src::HasSource,
};

#[salsa_macros::db]
#[derive(Clone)]
pub(crate) struct TestDB {
    storage: salsa::Storage<Self>,
    files: Arc<base_db::Files>,
    crates_map: Arc<CratesMap>,
    events: Arc<Mutex<Option<Vec<salsa::Event>>>>,
}

impl Default for TestDB {
    fn default() -> Self {
        let mut this = Self {
            storage: Default::default(),
            events: Default::default(),
            files: Default::default(),
            crates_map: Default::default(),
        };
        this.set_expand_proc_attr_macros_with_durability(true, Durability::HIGH);
        // This needs to be here otherwise `CrateGraphBuilder` panics.
        this.set_all_crates(Arc::new(Box::new([])));
        CrateGraphBuilder::default().set_in_db(&mut this);
        this
    }
}

#[salsa_macros::db]
impl salsa::Database for TestDB {
    fn salsa_event(&self, event: &dyn std::ops::Fn() -> salsa::Event) {
        let mut events = self.events.lock().unwrap();
        if let Some(events) = &mut *events {
            let event = event();
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

#[salsa_macros::db]
impl SourceDatabase for TestDB {
    fn file_text(&self, file_id: base_db::FileId) -> FileText {
        self.files.file_text(file_id)
    }

    fn set_file_text(&mut self, file_id: base_db::FileId, text: &str) {
        let files = Arc::clone(&self.files);
        files.set_file_text(self, file_id, text);
    }

    fn set_file_text_with_durability(
        &mut self,
        file_id: base_db::FileId,
        text: &str,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_file_text_with_durability(self, file_id, text, durability);
    }

    /// Source root of the file.
    fn source_root(&self, source_root_id: SourceRootId) -> SourceRootInput {
        self.files.source_root(source_root_id)
    }

    fn set_source_root_with_durability(
        &mut self,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_source_root_with_durability(self, source_root_id, source_root, durability);
    }

    fn file_source_root(&self, id: base_db::FileId) -> FileSourceRootInput {
        self.files.file_source_root(id)
    }

    fn set_file_source_root_with_durability(
        &mut self,
        id: base_db::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_file_source_root_with_durability(self, id, source_root_id, durability);
    }

    fn crates_map(&self) -> Arc<CratesMap> {
        self.crates_map.clone()
    }
}

impl TestDB {
    pub(crate) fn fetch_test_crate(&self) -> Crate {
        let all_crates = self.all_crates();
        all_crates
            .iter()
            .copied()
            .find(|&krate| {
                krate.extra_data(self).display_name.as_ref().map(|it| it.canonical_name().as_str())
                    == Some("ra_test_fixture")
            })
            .unwrap_or(*all_crates.last().unwrap())
    }

    pub(crate) fn module_for_file(&self, file_id: FileId) -> ModuleId {
        for &krate in self.relevant_crates(file_id).iter() {
            let crate_def_map = crate_def_map(self, krate);
            for (local_id, data) in crate_def_map.modules() {
                if data.origin.file_id().map(|file_id| file_id.file_id(self)) == Some(file_id) {
                    return crate_def_map.module_id(local_id);
                }
            }
        }
        panic!("Can't find module for file")
    }

    pub(crate) fn module_at_position(&self, position: FilePosition) -> ModuleId {
        let file_module = self.module_for_file(position.file_id.file_id(self));
        let mut def_map = file_module.def_map(self);
        let module = self.mod_at_position(def_map, position);

        def_map = match self.block_at_position(def_map, position) {
            Some(it) => it,
            None => return def_map.module_id(module),
        };
        loop {
            let new_map = self.block_at_position(def_map, position);
            match new_map {
                Some(new_block) if !std::ptr::eq(&new_block, &def_map) => {
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

    fn block_at_position(&self, def_map: &DefMap, position: FilePosition) -> Option<&DefMap> {
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
        let source_map = self.body_with_source_map(def_with_body).1;
        let scopes = self.expr_scopes(def_with_body);

        let root_syntax_node = self.parse(position.file_id).syntax_node();
        let scope_iter =
            algo::ancestors_at_offset(&root_syntax_node, position.offset).filter_map(|node| {
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

            if let Some(block) = containing_blocks.next().map(|block| block_def_map(self, block)) {
                return Some(block);
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
                    let ingredient = self
                        .as_dyn_database()
                        .ingredient_debug_name(database_key.ingredient_index());
                    Some(ingredient.to_string())
                }
                _ => None,
            })
            .collect()
    }
}
