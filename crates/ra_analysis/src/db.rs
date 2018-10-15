use std::{
    fmt,
    sync::Arc,
    hash::{Hash, Hasher},
};
use salsa;
use rustc_hash::FxHashSet;
use ra_syntax::File;
use ra_editor::{LineIndex};
use crate::{
    symbol_index::SymbolIndex,
    module_map::{ModulesDatabase, ModuleTreeQuery, ModuleDescriptorQuery},
    FileId, FileResolverImp,
};

#[derive(Default)]
pub(crate) struct RootDatabase {
    runtime: salsa::runtime::Runtime<RootDatabase>,
}

impl fmt::Debug for RootDatabase {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("RootDatabase { ... }")
    }
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::runtime::Runtime<RootDatabase> {
        &self.runtime
    }
}

salsa::database_storage! {
    pub(crate) struct RootDatabaseStorage for RootDatabase {
        impl FilesDatabase {
            fn file_text() for FileTextQuery;
            fn file_set() for FileSetQuery;
        }
        impl SyntaxDatabase {
            fn file_syntax() for FileSyntaxQuery;
            fn file_lines() for FileLinesQuery;
            fn file_symbols() for FileSymbolsQuery;
        }
        impl ModulesDatabase {
            fn module_tree() for ModuleTreeQuery;
            fn module_descriptor() for ModuleDescriptorQuery;
        }
    }
}

salsa::query_group! {
    pub(crate) trait FilesDatabase: salsa::Database {
        fn file_text(file_id: FileId) -> Arc<String> {
            type FileTextQuery;
            storage input;
        }
        fn file_set(key: ()) -> Arc<FileSet> {
            type FileSetQuery;
            storage input;
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct FileSet {
    pub(crate) files: FxHashSet<FileId>,
    pub(crate) resolver: FileResolverImp,
}

impl Hash for FileSet {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        let mut files = self.files.iter().cloned().collect::<Vec<_>>();
        files.sort();
        files.hash(hasher);
    }
}

salsa::query_group! {
    pub(crate) trait SyntaxDatabase: FilesDatabase {
        fn file_syntax(file_id: FileId) -> File {
            type FileSyntaxQuery;
        }
        fn file_lines(file_id: FileId) -> Arc<LineIndex> {
            type FileLinesQuery;
        }
        fn file_symbols(file_id: FileId) -> Arc<SymbolIndex> {
            type FileSymbolsQuery;
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
fn file_symbols(db: &impl SyntaxDatabase, file_id: FileId) -> Arc<SymbolIndex> {
    let syntax = db.file_syntax(file_id);
    Arc::new(SymbolIndex::for_file(file_id, syntax))
}
