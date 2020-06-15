//! Maps paths to compact integer ids. We don't care about clearings paths which
//! no longer exist -- the assumption is total size of paths we ever look at is
//! not too big.
use rustc_hash::FxHashMap;

use crate::{FileId, VfsPath};

#[derive(Default)]
pub(crate) struct PathInterner {
    map: FxHashMap<VfsPath, FileId>,
    vec: Vec<VfsPath>,
}

impl PathInterner {
    pub(crate) fn get(&self, path: &VfsPath) -> Option<FileId> {
        self.map.get(path).copied()
    }
    pub(crate) fn intern(&mut self, path: VfsPath) -> FileId {
        if let Some(id) = self.get(&path) {
            return id;
        }
        let id = FileId(self.vec.len() as u32);
        self.map.insert(path.clone(), id);
        self.vec.push(path);
        id
    }

    pub(crate) fn lookup(&self, id: FileId) -> &VfsPath {
        &self.vec[id.0 as usize]
    }
}
