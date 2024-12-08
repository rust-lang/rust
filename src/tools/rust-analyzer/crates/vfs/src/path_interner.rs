//! Maps paths to compact integer ids. We don't care about clearings paths which
//! no longer exist -- the assumption is total size of paths we ever look at is
//! not too big.
use std::hash::BuildHasherDefault;

use indexmap::IndexSet;
use rustc_hash::FxHasher;

use crate::{FileId, VfsPath};

/// Structure to map between [`VfsPath`] and [`FileId`].
#[derive(Default)]
pub(crate) struct PathInterner {
    map: IndexSet<VfsPath, BuildHasherDefault<FxHasher>>,
}

impl PathInterner {
    /// Get the id corresponding to `path`.
    ///
    /// If `path` does not exists in `self`, returns [`None`].
    pub(crate) fn get(&self, path: &VfsPath) -> Option<FileId> {
        self.map.get_index_of(path).map(|i| FileId(i as u32))
    }

    /// Insert `path` in `self`.
    ///
    /// - If `path` already exists in `self`, returns its associated id;
    /// - Else, returns a newly allocated id.
    pub(crate) fn intern(&mut self, path: VfsPath) -> FileId {
        let (id, _added) = self.map.insert_full(path);
        assert!(id < u32::MAX as usize);
        FileId(id as u32)
    }

    /// Returns the path corresponding to `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not exists in `self`.
    pub(crate) fn lookup(&self, id: FileId) -> &VfsPath {
        self.map.get_index(id.0 as usize).unwrap()
    }
}
