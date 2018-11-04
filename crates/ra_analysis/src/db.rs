use std::sync::Arc;

use ra_editor::LineIndex;
use ra_syntax::{File, SyntaxNode};
use salsa::{self, Database};

use crate::{
    db,
    descriptors::{
        DescriptorDatabase, FnScopesQuery, FnSyntaxQuery, ModuleScopeQuery, ModuleTreeQuery,
        SubmodulesQuery,
    },
    symbol_index::SymbolIndex,
    syntax_ptr::SyntaxPtr,
    Cancelable, Canceled, FileId,
};

#[derive(Debug)]
pub(crate) struct RootDatabase {
    runtime: salsa::Runtime<RootDatabase>,
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<RootDatabase> {
        &self.runtime
    }
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        let mut db = RootDatabase {
            runtime: Default::default(),
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

pub(crate) fn check_canceled(db: &impl salsa::Database) -> Cancelable<()> {
    if db.salsa_runtime().is_current_revision_canceled() {
        Err(Canceled)
    } else {
        Ok(())
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase {
            runtime: self.runtime.snapshot(self),
        })
    }
}

salsa::database_storage! {
    pub(crate) struct RootDatabaseStorage for RootDatabase {
        impl crate::input::FilesDatabase {
            fn file_text() for crate::input::FileTextQuery;
            fn file_source_root() for crate::input::FileSourceRootQuery;
            fn source_root() for crate::input::SourceRootQuery;
            fn libraries() for crate::input::LibrariesQuery;
            fn library_symbols() for crate::input::LibrarySymbolsQuery;
            fn crate_graph() for crate::input::CrateGraphQuery;
        }
        impl SyntaxDatabase {
            fn file_syntax() for FileSyntaxQuery;
            fn file_lines() for FileLinesQuery;
            fn file_symbols() for FileSymbolsQuery;
            fn resolve_syntax_ptr() for ResolveSyntaxPtrQuery;
        }
        impl DescriptorDatabase {
            fn module_tree() for ModuleTreeQuery;
            fn module_descriptor() for SubmodulesQuery;
            fn module_scope() for ModuleScopeQuery;
            fn fn_syntax() for FnSyntaxQuery;
            fn fn_scopes() for FnScopesQuery;
        }
    }
}

salsa::query_group! {
    pub(crate) trait SyntaxDatabase: crate::input::FilesDatabase {
        fn file_syntax(file_id: FileId) -> File {
            type FileSyntaxQuery;
        }
        fn file_lines(file_id: FileId) -> Arc<LineIndex> {
            type FileLinesQuery;
        }
        fn file_symbols(file_id: FileId) -> Cancelable<Arc<SymbolIndex>> {
            type FileSymbolsQuery;
        }
        fn resolve_syntax_ptr(ptr: SyntaxPtr) -> SyntaxNode {
            type ResolveSyntaxPtrQuery;
            // Don't retain syntax trees in memory
            storage volatile;
            use fn crate::syntax_ptr::resolve_syntax_ptr;
        }
    }
}

fn file_syntax(db: &impl SyntaxDatabase, file_id: FileId) -> File {
    let text = db.file_text(file_id);
    File::parse(&*text)
}
fn file_lines(db: &impl SyntaxDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}
fn file_symbols(db: &impl SyntaxDatabase, file_id: FileId) -> Cancelable<Arc<SymbolIndex>> {
    db::check_canceled(db)?;
    let syntax = db.file_syntax(file_id);
    Ok(Arc::new(SymbolIndex::for_file(file_id, syntax)))
}
