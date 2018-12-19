use std::sync::Arc;

use parking_lot::Mutex;
use salsa::{self, Database};
use ra_db::{LocationIntener, BaseDatabase, FilePosition, FileId, WORKSPACE, CrateGraph, SourceRoot};
use relative_path::RelativePathBuf;
use test_utils::{parse_fixture, CURSOR_MARKER, extract_offset};

use crate::{db, DefId, DefLoc};

#[derive(Debug)]
pub(crate) struct MockDatabase {
    events: Mutex<Option<Vec<salsa::Event<MockDatabase>>>>,
    runtime: salsa::Runtime<MockDatabase>,
    id_maps: Arc<IdMaps>,
}

impl MockDatabase {
    pub(crate) fn with_files(fixture: &str) -> (MockDatabase, SourceRoot) {
        let (db, source_root, position) = MockDatabase::from_fixture(fixture);
        assert!(position.is_none());
        (db, source_root)
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

        let mut position = None;
        let mut source_root = SourceRoot::default();
        for entry in parse_fixture(fixture) {
            if entry.text.contains(CURSOR_MARKER) {
                assert!(
                    position.is_none(),
                    "only one marker (<|>) per fixture is allowed"
                );
                position =
                    Some(db.add_file_with_position(&mut source_root, &entry.meta, &entry.text));
            } else {
                db.add_file(&mut source_root, &entry.meta, &entry.text);
            }
        }
        db.query_mut(ra_db::SourceRootQuery)
            .set(WORKSPACE, Arc::new(source_root.clone()));
        (db, source_root, position)
    }

    fn add_file(&mut self, source_root: &mut SourceRoot, path: &str, text: &str) -> FileId {
        assert!(path.starts_with('/'));
        let path = RelativePathBuf::from_path(&path[1..]).unwrap();
        let file_id = FileId(source_root.files.len() as u32);
        let text = Arc::new(text.to_string());
        self.query_mut(ra_db::FileTextQuery).set(file_id, text);
        self.query_mut(ra_db::FileRelativePathQuery)
            .set(file_id, path.clone());
        self.query_mut(ra_db::FileSourceRootQuery)
            .set(file_id, WORKSPACE);
        source_root.files.insert(path, file_id);
        file_id
    }

    fn add_file_with_position(
        &mut self,
        source_root: &mut SourceRoot,
        path: &str,
        text: &str,
    ) -> FilePosition {
        let (offset, text) = extract_offset(text);
        let file_id = self.add_file(source_root, path, &text);
        FilePosition { file_id, offset }
    }
}

#[derive(Debug, Default)]
struct IdMaps {
    defs: LocationIntener<DefLoc, DefId>,
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
            id_maps: Default::default(),
        };
        db.query_mut(ra_db::SourceRootQuery)
            .set(ra_db::WORKSPACE, Default::default());
        db.query_mut(ra_db::CrateGraphQuery)
            .set((), Default::default());
        db.query_mut(ra_db::LibrariesQuery)
            .set((), Default::default());
        db
    }
}

impl salsa::ParallelDatabase for MockDatabase {
    fn snapshot(&self) -> salsa::Snapshot<MockDatabase> {
        salsa::Snapshot::new(MockDatabase {
            events: Default::default(),
            runtime: self.runtime.snapshot(self),
            id_maps: self.id_maps.clone(),
        })
    }
}

impl BaseDatabase for MockDatabase {}

impl AsRef<LocationIntener<DefLoc, DefId>> for MockDatabase {
    fn as_ref(&self) -> &LocationIntener<DefLoc, DefId> {
        &self.id_maps.defs
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
            fn libraries() for ra_db::LibrariesQuery;
            fn crate_graph() for ra_db::CrateGraphQuery;
        }
        impl ra_db::SyntaxDatabase {
            fn source_file() for ra_db::SourceFileQuery;
            fn file_lines() for ra_db::FileLinesQuery;
        }
        impl db::HirDatabase {
            fn module_tree() for db::ModuleTreeQuery;
            fn fn_scopes() for db::FnScopesQuery;
            fn file_items() for db::SourceFileItemsQuery;
            fn file_item() for db::FileItemQuery;
            fn input_module_items() for db::InputModuleItemsQuery;
            fn item_map() for db::ItemMapQuery;
            fn fn_syntax() for db::FnSyntaxQuery;
            fn submodules() for db::SubmodulesQuery;
        }
    }
}
