use std::{fmt, sync::Arc};
use salsa::{self, Database};
use ra_db::{LocationIntener, BaseDatabase};

use crate::{
    symbol_index,
};

#[derive(Debug)]
pub(crate) struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
    id_maps: Arc<IdMaps>,
}

#[derive(Default)]
struct IdMaps {
    defs: LocationIntener<hir::DefLoc, hir::DefId>,
    macros: LocationIntener<hir::MacroCallLoc, hir::MacroCallId>,
}

impl fmt::Debug for IdMaps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("IdMaps")
            .field("n_defs", &self.defs.len())
            .finish()
    }
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<RootDatabase> {
        &self.runtime
    }
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        let mut db = RootDatabase {
            runtime: salsa::Runtime::default(),
            id_maps: Default::default(),
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

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            runtime: self.runtime.snapshot(self),
            id_maps: self.id_maps.clone(),
        })
    }
}

impl BaseDatabase for RootDatabase {}

impl AsRef<LocationIntener<hir::DefLoc, hir::DefId>> for RootDatabase {
    fn as_ref(&self) -> &LocationIntener<hir::DefLoc, hir::DefId> {
        &self.id_maps.defs
    }
}

impl AsRef<LocationIntener<hir::MacroCallLoc, hir::MacroCallId>> for RootDatabase {
    fn as_ref(&self) -> &LocationIntener<hir::MacroCallLoc, hir::MacroCallId> {
        &self.id_maps.macros
    }
}

salsa::database_storage! {
    pub(crate) struct RootDatabaseStorage for RootDatabase {
        impl ra_db::FilesDatabase {
            fn file_text() for ra_db::FileTextQuery;
            fn file_relative_path() for ra_db::FileRelativePathQuery;
            fn file_source_root() for ra_db::FileSourceRootQuery;
            fn source_root() for ra_db::SourceRootQuery;
            fn local_roots() for ra_db::LocalRootsQuery;
            fn library_roots() for ra_db::LibraryRootsQuery;
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
            fn hir_source_file() for hir::db::HirSourceFileQuery;
            fn expand_macro_invocation() for hir::db::ExpandMacroCallQuery;
            fn module_tree() for hir::db::ModuleTreeQuery;
            fn fn_scopes() for hir::db::FnScopesQuery;
            fn file_items() for hir::db::SourceFileItemsQuery;
            fn file_item() for hir::db::FileItemQuery;
            fn input_module_items() for hir::db::InputModuleItemsQuery;
            fn item_map() for hir::db::ItemMapQuery;
            fn submodules() for hir::db::SubmodulesQuery;
            fn infer() for hir::db::InferQuery;
            fn type_for_def() for hir::db::TypeForDefQuery;
            fn type_for_field() for hir::db::TypeForFieldQuery;
            fn struct_data() for hir::db::StructDataQuery;
            fn enum_data() for hir::db::EnumDataQuery;
        }
    }
}
