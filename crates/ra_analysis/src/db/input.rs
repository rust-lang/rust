use std::{
    sync::Arc,
    hash::{Hasher, Hash},
};

use salsa;
use rustc_hash::FxHashSet;

use crate::{FileId, FileResolverImp, CrateGraph, symbol_index::SymbolIndex};

salsa::query_group! {
    pub(crate) trait FilesDatabase: salsa::Database {
        fn file_text(file_id: FileId) -> Arc<String> {
            type FileTextQuery;
            storage input;
        }
        fn file_source_root(file_id: FileId) -> SourceRootId {
            type FileSourceRootQuery;
            storage input;
        }
        fn source_root(id: SourceRootId) -> Arc<SourceRoot> {
            type SourceRootQuery;
            storage input;
        }
        fn libraries() -> Arc<Vec<SourceRootId>> {
            type LibrarieseQuery;
            storage input;
        }
        fn library_symbols(id: SourceRootId) -> Arc<SymbolIndex> {
            type LibrarySymbolsQuery;
            storage input;
        }
        fn crate_graph() -> Arc<CrateGraph> {
            type CrateGraphQuery;
            storage input;
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SourceRootId(pub(crate) u32);

#[derive(Clone, Default, Debug, Eq)]
pub(crate) struct SourceRoot {
    pub(crate) file_resolver: FileResolverImp,
    pub(crate) files: FxHashSet<FileId>,
}

impl PartialEq for SourceRoot {
    fn eq(&self, other: &SourceRoot) -> bool {
        self.file_resolver == other.file_resolver
    }
}

impl Hash for SourceRoot {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.file_resolver.hash(hasher);
    }
}

pub(crate) const WORKSPACE: SourceRootId = SourceRootId(0);


#[derive(Default, Debug, Eq)]
pub(crate) struct FileSet {
    pub(crate) files: FxHashSet<FileId>,
    pub(crate) resolver: FileResolverImp,
}

impl PartialEq for FileSet {
    fn eq(&self, other: &FileSet) -> bool {
        self.files == other.files && self.resolver == other.resolver
    }
}

impl Hash for FileSet {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        let mut files = self.files.iter().cloned().collect::<Vec<_>>();
        files.sort();
        files.hash(hasher);
    }
}

