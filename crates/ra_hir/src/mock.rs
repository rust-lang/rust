use std::{sync::Arc, panic};

use parking_lot::Mutex;
use ra_db::{
    BaseDatabase, FilePosition, FileId, CrateGraph, SourceRoot, SourceRootId,
    salsa::{self, Database},
};
use relative_path::RelativePathBuf;
use test_utils::{parse_fixture, CURSOR_MARKER, extract_offset};

use crate::{db, HirInterner};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

#[derive(Debug)]
pub(crate) struct MockDatabase {
    events: Mutex<Option<Vec<salsa::Event<MockDatabase>>>>,
    runtime: salsa::Runtime<MockDatabase>,
    interner: Arc<HirInterner>,
    file_counter: u32,
}

impl panic::RefUnwindSafe for MockDatabase {}

impl MockDatabase {
    pub(crate) fn with_files(fixture: &str) -> (MockDatabase, SourceRoot) {
        let (db, source_root, position) = MockDatabase::from_fixture(fixture);
        assert!(position.is_none());
        (db, source_root)
    }

    pub(crate) fn with_single_file(text: &str) -> (MockDatabase, SourceRoot, FileId) {
        let mut db = MockDatabase::default();
        let mut source_root = SourceRoot::default();
        let file_id = db.add_file(WORKSPACE, &mut source_root, "/main.rs", text);
        db.query_mut(ra_db::SourceRootQuery)
            .set(WORKSPACE, Arc::new(source_root.clone()));
        (db, source_root, file_id)
    }

    pub(crate) fn with_position(fixture: &str) -> (MockDatabase, FilePosition) {
        let (db, _, position) = MockDatabase::from_fixture(fixture);
        let position = position.expect("expected a marker ( <|> )");
        (db, position)
    }

    pub(crate) fn set_crate_graph(&mut self, crate_graph: CrateGraph) {
        self.query_mut(ra_db::CrateGraphQuery)
            .set((), Arc::new(crate_graph));
    }

    fn from_fixture(fixture: &str) -> (MockDatabase, SourceRoot, Option<FilePosition>) {
        let mut db = MockDatabase::default();

        let (source_root, pos) = db.add_fixture(WORKSPACE, fixture);

        (db, source_root, pos)
    }

    pub fn add_fixture(
        &mut self,
        source_root_id: SourceRootId,
        fixture: &str,
    ) -> (SourceRoot, Option<FilePosition>) {
        let mut position = None;
        let mut source_root = SourceRoot::default();
        for entry in parse_fixture(fixture) {
            if entry.text.contains(CURSOR_MARKER) {
                assert!(
                    position.is_none(),
                    "only one marker (<|>) per fixture is allowed"
                );
                position = Some(self.add_file_with_position(
                    source_root_id,
                    &mut source_root,
                    &entry.meta,
                    &entry.text,
                ));
            } else {
                self.add_file(source_root_id, &mut source_root, &entry.meta, &entry.text);
            }
        }
        self.query_mut(ra_db::SourceRootQuery)
            .set(source_root_id, Arc::new(source_root.clone()));
        (source_root, position)
    }

    fn add_file(
        &mut self,
        source_root_id: SourceRootId,
        source_root: &mut SourceRoot,
        path: &str,
        text: &str,
    ) -> FileId {
        assert!(path.starts_with('/'));
        let is_crate_root = path == "/lib.rs" || path == "/main.rs";

        let path = RelativePathBuf::from_path(&path[1..]).unwrap();
        let file_id = FileId(self.file_counter);
        self.file_counter += 1;
        let text = Arc::new(text.to_string());
        self.query_mut(ra_db::FileTextQuery).set(file_id, text);
        self.query_mut(ra_db::FileRelativePathQuery)
            .set(file_id, path.clone());
        self.query_mut(ra_db::FileSourceRootQuery)
            .set(file_id, source_root_id);
        source_root.files.insert(path, file_id);

        if is_crate_root {
            let mut crate_graph = CrateGraph::default();
            crate_graph.add_crate_root(file_id);
            self.set_crate_graph(crate_graph);
        }
        file_id
    }

    fn add_file_with_position(
        &mut self,
        source_root_id: SourceRootId,
        source_root: &mut SourceRoot,
        path: &str,
        text: &str,
    ) -> FilePosition {
        let (offset, text) = extract_offset(text);
        let file_id = self.add_file(source_root_id, source_root, path, &text);
        FilePosition { file_id, offset }
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
        let mut db = MockDatabase {
            events: Default::default(),
            runtime: salsa::Runtime::default(),
            interner: Default::default(),
            file_counter: 0,
        };
        db.query_mut(ra_db::CrateGraphQuery)
            .set((), Default::default());
        db.query_mut(ra_db::LocalRootsQuery)
            .set((), Default::default());
        db.query_mut(ra_db::LibraryRootsQuery)
            .set((), Default::default());
        db
    }
}

impl salsa::ParallelDatabase for MockDatabase {
    fn snapshot(&self) -> salsa::Snapshot<MockDatabase> {
        salsa::Snapshot::new(MockDatabase {
            events: Default::default(),
            runtime: self.runtime.snapshot(self),
            interner: Arc::clone(&self.interner),
            file_counter: self.file_counter,
        })
    }
}

impl BaseDatabase for MockDatabase {}

impl AsRef<HirInterner> for MockDatabase {
    fn as_ref(&self) -> &HirInterner {
        &self.interner
    }
}

impl MockDatabase {
    pub(crate) fn log(&self, f: impl FnOnce()) -> Vec<salsa::Event<MockDatabase>> {
        *self.events.lock() = Some(Vec::new());
        f();
        let events = self.events.lock().take().unwrap();
        events
    }

    pub(crate) fn log_executed(&self, f: impl FnOnce()) -> Vec<String> {
        let events = self.log(f);
        events
            .into_iter()
            .filter_map(|e| match e.kind {
                // This pretty horrible, but `Debug` is the only way to inspect
                // QueryDescriptor at the moment.
                salsa::EventKind::WillExecute { descriptor } => Some(format!("{:?}", descriptor)),
                _ => None,
            })
            .collect()
    }
}

salsa::database_storage! {
    pub(crate) struct MockDatabaseStorage for MockDatabase {
        impl ra_db::FilesDatabase {
            fn file_text() for ra_db::FileTextQuery;
            fn file_relative_path() for ra_db::FileRelativePathQuery;
            fn file_source_root() for ra_db::FileSourceRootQuery;
            fn source_root() for ra_db::SourceRootQuery;
            fn source_root_crates() for ra_db::SourceRootCratesQuery;
            fn local_roots() for ra_db::LocalRootsQuery;
            fn library_roots() for ra_db::LibraryRootsQuery;
            fn crate_graph() for ra_db::CrateGraphQuery;
        }
        impl ra_db::SyntaxDatabase {
            fn source_file() for ra_db::SourceFileQuery;
        }
        impl db::HirDatabase {
            fn hir_source_file() for db::HirSourceFileQuery;
            fn expand_macro_invocation() for db::ExpandMacroInvocationQuery;
            fn module_tree() for db::ModuleTreeQuery;
            fn fn_scopes() for db::FnScopesQuery;
            fn file_items() for db::FileItemsQuery;
            fn file_item() for db::FileItemQuery;
            fn lower_module() for db::LowerModuleQuery;
            fn lower_module_module() for db::LowerModuleModuleQuery;
            fn lower_module_source_map() for db::LowerModuleSourceMapQuery;
            fn item_map() for db::ItemMapQuery;
            fn submodules() for db::SubmodulesQuery;
            fn infer() for db::InferQuery;
            fn type_for_def() for db::TypeForDefQuery;
            fn type_for_field() for db::TypeForFieldQuery;
            fn struct_data() for db::StructDataQuery;
            fn enum_data() for db::EnumDataQuery;
            fn impls_in_module() for db::ImplsInModuleQuery;
            fn impls_in_crate() for db::ImplsInCrateQuery;
            fn body_hir() for db::BodyHirQuery;
            fn body_syntax_mapping() for db::BodySyntaxMappingQuery;
            fn fn_signature() for db::FnSignatureQuery;
            fn generic_params() for db::GenericParamsQuery;
        }
    }
}
