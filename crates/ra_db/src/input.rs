use std::sync::Arc;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use salsa;

use crate::file_resolver::FileResolverImp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateId(pub u32);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrateGraph {
    pub(crate) crate_roots: FxHashMap<CrateId, FileId>,
}

impl CrateGraph {
    pub fn crate_root(&self, crate_id: CrateId) -> FileId {
        self.crate_roots[&crate_id]
    }
    pub fn add_crate_root(&mut self, file_id: FileId) -> CrateId {
        let crate_id = CrateId(self.crate_roots.len() as u32);
        let prev = self.crate_roots.insert(crate_id, file_id);
        assert!(prev.is_none());
        crate_id
    }
    pub fn crate_id_for_crate_root(&self, file_id: FileId) -> Option<CrateId> {
        let (&crate_id, _) = self
            .crate_roots
            .iter()
            .find(|(_crate_id, &root_id)| root_id == file_id)?;
        Some(crate_id)
    }
}

salsa::query_group! {
    pub trait FilesDatabase: salsa::Database {
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
            type LibrariesQuery;
            storage input;
        }
        fn crate_graph() -> Arc<CrateGraph> {
            type CrateGraphQuery;
            storage input;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRootId(pub u32);

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SourceRoot {
    pub file_resolver: FileResolverImp,
    pub files: FxHashSet<FileId>,
}

pub const WORKSPACE: SourceRootId = SourceRootId(0);
