use std::{
    sync::Arc,
    fmt,
};

use salsa;
use rustc_hash::FxHashSet;
use relative_path::RelativePath;
use rustc_hash::FxHashMap;

use crate::{symbol_index::SymbolIndex, FileResolverImp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateId(pub u32);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrateGraph {
    pub(crate) crate_roots: FxHashMap<CrateId, FileId>,
}

impl CrateGraph {
    pub fn new() -> CrateGraph {
        CrateGraph::default()
    }
    pub fn add_crate_root(&mut self, file_id: FileId) -> CrateId{
        let crate_id = CrateId(self.crate_roots.len() as u32);
        let prev = self.crate_roots.insert(crate_id, file_id);
        assert!(prev.is_none());
        crate_id
    }
}

pub trait FileResolver: fmt::Debug + Send + Sync + 'static {
    fn file_stem(&self, file_id: FileId) -> String;
    fn resolve(&self, file_id: FileId, path: &RelativePath) -> Option<FileId>;
}

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

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct SourceRoot {
    pub(crate) file_resolver: FileResolverImp,
    pub(crate) files: FxHashSet<FileId>,
}

pub(crate) const WORKSPACE: SourceRootId = SourceRootId(0);
