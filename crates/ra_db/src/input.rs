use std::sync::Arc;

use rustc_hash::{FxHashMap};
use relative_path::RelativePathBuf;
use ra_syntax::SmolStr;
use salsa;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRootId(pub u32);

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
    dependencies: Vec<Dependency>,
}

impl CrateData {
    fn new(file_id: FileId) -> CrateData {
        CrateData {
            file_id,
            dependencies: Vec::new(),
        }
    }

    fn add_dep(&mut self, name: SmolStr, crate_id: CrateId) {
        self.dependencies.push(Dependency { name, crate_id })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    pub crate_id: CrateId,
    pub name: SmolStr,
}

impl Dependency {
    pub fn crate_id(&self) -> CrateId {
        self.crate_id
    }
}

impl CrateGraph {
    pub fn add_crate_root(&mut self, file_id: FileId) -> CrateId {
        let crate_id = CrateId(self.arena.len() as u32);
        let prev = self.arena.insert(crate_id, CrateData::new(file_id));
        assert!(prev.is_none());
        crate_id
    }
    //FIXME: check that we don't have cycles here.
    // Just a simple depth first search from `to` should work,
    // the graph is small.
    pub fn add_dep(&mut self, from: CrateId, name: SmolStr, to: CrateId) {
        self.arena.get_mut(&from).unwrap().add_dep(name, to)
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
    pub fn dependencies<'a>(
        &'a self,
        crate_id: CrateId,
    ) -> impl Iterator<Item = &'a Dependency> + 'a {
        self.arena[&crate_id].dependencies.iter()
    }
}

salsa::query_group! {
    pub trait FilesDatabase: salsa::Database {
        fn file_text(file_id: FileId) -> Arc<String> {
            type FileTextQuery;
            storage input;
        }
        /// Path to a file, relative to the root of its source root.
        fn file_relative_path(file_id: FileId) -> RelativePathBuf {
            type FileRelativePathQuery;
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
        fn local_roots() -> Arc<Vec<SourceRootId>> {
            type LocalRootsQuery;
            storage input;
        }
        fn library_roots() -> Arc<Vec<SourceRootId>> {
            type LibraryRootsQuery;
            storage input;
        }
        fn crate_graph() -> Arc<CrateGraph> {
            type CrateGraphQuery;
            storage input;
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SourceRoot {
    pub files: FxHashMap<RelativePathBuf, FileId>,
}
