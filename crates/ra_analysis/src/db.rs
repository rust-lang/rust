use std::sync::Arc;
#[cfg(test)]
use parking_lot::Mutex;
use salsa::{self, Database};
use ra_db::{LocationIntener, BaseDatabase};

use crate::{
    hir::{self, DefId, DefLoc, FnId, SourceItemId},
    symbol_index,
};

#[derive(Debug)]
pub(crate) struct RootDatabase {
    #[cfg(test)]
    events: Mutex<Option<Vec<salsa::Event<RootDatabase>>>>,
    #[cfg(not(test))]
    events: (),

    runtime: salsa::Runtime<RootDatabase>,
    id_maps: Arc<IdMaps>,
}

#[derive(Debug, Default)]
struct IdMaps {
    fns: LocationIntener<SourceItemId, FnId>,
    defs: LocationIntener<DefLoc, DefId>,
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

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            events: Default::default(),
            runtime: self.runtime.snapshot(self),
            id_maps: self.id_maps.clone(),
        })
    }
}

impl BaseDatabase for RootDatabase {}

impl AsRef<LocationIntener<DefLoc, DefId>> for RootDatabase {
    fn as_ref(&self) -> &LocationIntener<DefLoc, DefId> {
        &self.id_maps.defs
    }
}

impl AsRef<LocationIntener<hir::SourceItemId, FnId>> for RootDatabase {
    fn as_ref(&self) -> &LocationIntener<hir::SourceItemId, FnId> {
        &self.id_maps.fns
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
        impl ra_db::FilesDatabase {
            fn file_text() for ra_db::FileTextQuery;
            fn file_source_root() for ra_db::FileSourceRootQuery;
            fn source_root() for ra_db::SourceRootQuery;
            fn libraries() for ra_db::LibrariesQuery;
            fn crate_graph() for ra_db::CrateGraphQuery;
        }
        impl ra_db::SyntaxDatabase {
            fn source_file() for ra_db::SourceFileQuery;
            fn file_lines() for ra_db::FileLinesQuery;
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
