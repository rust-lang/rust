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
    arena: FxHashMap<CrateId, CrateData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CrateData {
    file_id: FileId,
    deps: Vec<Dependency>,
}

impl CrateData {
    fn new(file_id: FileId) -> CrateData {
        CrateData {
            file_id,
            deps: Vec::new(),
        }
    }

    fn add_dep(&mut self, dep: CrateId) {
        self.deps.push(Dependency { crate_: dep })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    crate_: CrateId,
}

impl CrateGraph {
    pub fn add_crate_root(&mut self, file_id: FileId) -> CrateId {
        let crate_id = CrateId(self.arena.len() as u32);
        let prev = self.arena.insert(crate_id, CrateData::new(file_id));
        assert!(prev.is_none());
        crate_id
    }
    pub fn add_dep(&mut self, from: CrateId, to: CrateId) {
        self.arena.get_mut(&from).unwrap().add_dep(to)
    }
    pub fn crate_root(&self, crate_id: CrateId) -> FileId {
        self.arena[&crate_id].file_id
    }
    pub fn crate_id_for_crate_root(&self, file_id: FileId) -> Option<CrateId> {
        let (&crate_id, _) = self
            .arena
            .iter()
            .find(|(_crate_id, data)| data.file_id == file_id)?;
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
