use std::sync::Arc;
#[cfg(test)]
use parking_lot::Mutex;
use ra_editor::LineIndex;
use ra_syntax::{SourceFileNode};
use salsa::{self, Database};

use crate::{
    hir,
    symbol_index,
    loc2id::{IdMaps},
    Cancelable, Canceled, FileId,
};

#[derive(Debug)]
pub(crate) struct RootDatabase {
    #[cfg(test)]
    events: Mutex<Option<Vec<salsa::Event<RootDatabase>>>>,
    #[cfg(not(test))]
    events: (),

    runtime: salsa::Runtime<RootDatabase>,
    id_maps: IdMaps,
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<RootDatabase> {
        &self.runtime
    }

    #[allow(unused)]
    fn salsa_event(&self, event: impl Fn() -> salsa::Event<RootDatabase>) {
        #[cfg(test)]
        {
            let mut events = self.events.lock();
            if let Some(events) = &mut *events {
                events.push(event());
            }
        }
    }
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        let mut db = RootDatabase {
            events: Default::default(),
            runtime: salsa::Runtime::default(),
            id_maps: IdMaps::default(),
        };
        db.query_mut(crate::input::SourceRootQuery)
            .set(crate::input::WORKSPACE, Default::default());
        db.query_mut(crate::input::CrateGraphQuery)
            .set((), Default::default());
        db.query_mut(crate::input::LibrariesQuery)
            .set((), Default::default());
        db
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            events: Default::default(),
            runtime: self.runtime.snapshot(self),
            id_maps: self.id_maps.clone(),
        })
    }
}

pub(crate) trait BaseDatabase: salsa::Database {
    fn id_maps(&self) -> &IdMaps;
    fn check_canceled(&self) -> Cancelable<()> {
        if self.salsa_runtime().is_current_revision_canceled() {
            Err(Canceled)
        } else {
            Ok(())
        }
    }
}

impl BaseDatabase for RootDatabase {
    fn id_maps(&self) -> &IdMaps {
        &self.id_maps
    }
}

#[cfg(test)]
impl RootDatabase {
    pub(crate) fn log(&self, f: impl FnOnce()) -> Vec<salsa::Event<RootDatabase>> {
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
    pub(crate) struct RootDatabaseStorage for RootDatabase {
        impl crate::input::FilesDatabase {
            fn file_text() for crate::input::FileTextQuery;
            fn file_source_root() for crate::input::FileSourceRootQuery;
            fn source_root() for crate::input::SourceRootQuery;
            fn libraries() for crate::input::LibrariesQuery;
            fn crate_graph() for crate::input::CrateGraphQuery;
        }
        impl SyntaxDatabase {
            fn file_syntax() for FileSyntaxQuery;
            fn file_lines() for FileLinesQuery;
        }
        impl symbol_index::SymbolsDatabase {
            fn file_symbols() for symbol_index::FileSymbolsQuery;
            fn library_symbols() for symbol_index::LibrarySymbolsQuery;
        }
        impl hir::db::HirDatabase {
            fn module_tree() for hir::db::ModuleTreeQuery;
            fn fn_scopes() for hir::db::FnScopesQuery;
            fn file_items() for hir::db::SourceFileItemsQuery;
            fn file_item() for hir::db::FileItemQuery;
            fn input_module_items() for hir::db::InputModuleItemsQuery;
            fn item_map() for hir::db::ItemMapQuery;
            fn fn_syntax() for hir::db::FnSyntaxQuery;
            fn submodules() for hir::db::SubmodulesQuery;
        }
    }
}

salsa::query_group! {
    pub(crate) trait SyntaxDatabase: crate::input::FilesDatabase + BaseDatabase {
        fn file_syntax(file_id: FileId) -> SourceFileNode {
            type FileSyntaxQuery;
        }
        fn file_lines(file_id: FileId) -> Arc<LineIndex> {
            type FileLinesQuery;
        }
    }
}

fn file_syntax(db: &impl SyntaxDatabase, file_id: FileId) -> SourceFileNode {
    let text = db.file_text(file_id);
    SourceFileNode::parse(&*text)
}
fn file_lines(db: &impl SyntaxDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}
