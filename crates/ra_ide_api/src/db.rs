use std::sync::Arc;

use ra_db::{
    BaseDatabase, FileId, Canceled,
    salsa::{self, Database},
};

use crate::{symbol_index, LineIndex};

#[derive(Debug)]
pub(crate) struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
    interner: Arc<hir::HirInterner>,
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<RootDatabase> {
        &self.runtime
    }
    fn on_propagated_panic(&self) -> ! {
        Canceled::throw()
    }
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        let mut db = RootDatabase {
            runtime: salsa::Runtime::default(),
            interner: Default::default(),
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
            interner: Arc::clone(&self.interner),
        })
    }
}

impl BaseDatabase for RootDatabase {}

impl AsRef<hir::HirInterner> for RootDatabase {
    fn as_ref(&self) -> &hir::HirInterner {
        &self.interner
    }
}

#[salsa::query_group]
pub(crate) trait LineIndexDatabase: ra_db::FilesDatabase + BaseDatabase {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &impl ra_db::FilesDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}

salsa::database_storage! {
    pub(crate) struct RootDatabaseStorage for RootDatabase {
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
        impl LineIndexDatabase {
            fn line_index() for LineIndexQuery;
        }
        impl symbol_index::SymbolsDatabase {
            fn file_symbols() for symbol_index::FileSymbolsQuery;
            fn library_symbols() for symbol_index::LibrarySymbolsQuery;
        }
        impl hir::db::HirDatabase {
            fn hir_source_file() for hir::db::HirSourceFileQuery;
            fn expand_macro_invocation() for hir::db::ExpandMacroInvocationQuery;
            fn module_tree() for hir::db::ModuleTreeQuery;
            fn fn_scopes() for hir::db::FnScopesQuery;
            fn file_items() for hir::db::FileItemsQuery;
            fn file_item() for hir::db::FileItemQuery;
            fn lower_module() for hir::db::LowerModuleQuery;
            fn lower_module_module() for hir::db::LowerModuleModuleQuery;
            fn lower_module_source_map() for hir::db::LowerModuleSourceMapQuery;
            fn item_map() for hir::db::ItemMapQuery;
            fn submodules() for hir::db::SubmodulesQuery;
            fn infer() for hir::db::InferQuery;
            fn type_for_def() for hir::db::TypeForDefQuery;
            fn type_for_field() for hir::db::TypeForFieldQuery;
            fn struct_data() for hir::db::StructDataQuery;
            fn enum_data() for hir::db::EnumDataQuery;
            fn impls_in_module() for hir::db::ImplsInModuleQuery;
            fn impls_in_crate() for hir::db::ImplsInCrateQuery;
            fn body_hir() for hir::db::BodyHirQuery;
            fn body_syntax_mapping() for hir::db::BodySyntaxMappingQuery;
            fn fn_signature() for hir::db::FnSignatureQuery;
            fn generic_params() for hir::db::GenericParamsQuery;
        }
    }
}
