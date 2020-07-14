//! # Virtual File System
//!
//! VFS stores all files read by rust-analyzer. Reading file contents from VFS
//! always returns the same contents, unless VFS was explicitly modified with
//! `set_file_contents`. All changes to VFS are logged, and can be retrieved via
//! `take_changes` method. The pack of changes is then pushed to `salsa` and
//! triggers incremental recomputation.
//!
//! Files in VFS are identified with `FileId`s -- interned paths. The notion of
//! the path, `VfsPath` is somewhat abstract: at the moment, it is represented
//! as an `std::path::PathBuf` internally, but this is an implementation detail.
//!
//! VFS doesn't do IO or file watching itself. For that, see the `loader`
//! module. `loader::Handle` is an object-safe trait which abstracts both file
//! loading and file watching. `Handle` is dynamically configured with a set of
//! directory entries which should be scanned and watched. `Handle` then
//! asynchronously pushes file changes. Directory entries are configured in
//! free-form via list of globs, it's up to the `Handle` to interpret the globs
//! in any specific way.
//!
//! A simple `WalkdirLoaderHandle` is provided, which doesn't implement watching
//! and just scans the directory using walkdir.
//!
//! VFS stores a flat list of files. `FileSet` can partition this list of files
//! into disjoint sets of files. Traversal-like operations (including getting
//! the neighbor file by the relative path) are handled by the `FileSet`.
//! `FileSet`s are also pushed to salsa and cause it to re-check `mod foo;`
//! declarations when files are created or deleted.
//!
//! `file_set::FileSet` and `loader::Entry` play similar, but different roles.
//! Both specify the "set of paths/files", one is geared towards file watching,
//! the other towards salsa changes. In particular, single `file_set::FileSet`
//! may correspond to several `loader::Entry`. For example, a crate from
//! crates.io which uses code generation would have two `Entries` -- for sources
//! in `~/.cargo`, and for generated code in `./target/debug/build`. It will
//! have a single `FileSet` which unions the two sources.
mod vfs_path;
mod path_interner;
pub mod file_set;
pub mod loader;

use std::{fmt, mem};

use crate::path_interner::PathInterner;

pub use crate::vfs_path::VfsPath;
pub use paths::{AbsPath, AbsPathBuf};

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct FileId(pub u32);

#[derive(Default)]
pub struct Vfs {
    interner: PathInterner,
    data: Vec<Option<Vec<u8>>>,
    changes: Vec<ChangedFile>,
}

pub struct ChangedFile {
    pub file_id: FileId,
    pub change_kind: ChangeKind,
}

impl ChangedFile {
    pub fn exists(&self) -> bool {
        self.change_kind != ChangeKind::Delete
    }
    pub fn is_created_or_deleted(&self) -> bool {
        matches!(self.change_kind, ChangeKind::Create | ChangeKind::Delete)
    }
}

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum ChangeKind {
    Create,
    Modify,
    Delete,
}

impl Vfs {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn file_id(&self, path: &VfsPath) -> Option<FileId> {
        self.interner.get(path).filter(|&it| self.get(it).is_some())
    }
    pub fn file_path(&self, file_id: FileId) -> VfsPath {
        self.interner.lookup(file_id).clone()
    }
    pub fn file_contents(&self, file_id: FileId) -> &[u8] {
        self.get(file_id).as_deref().unwrap()
    }
    pub fn iter(&self) -> impl Iterator<Item = (FileId, &VfsPath)> + '_ {
        (0..self.data.len())
            .map(|it| FileId(it as u32))
            .filter(move |&file_id| self.get(file_id).is_some())
            .map(move |file_id| {
                let path = self.interner.lookup(file_id);
                (file_id, path)
            })
    }
    pub fn set_file_contents(&mut self, path: VfsPath, contents: Option<Vec<u8>>) {
        let file_id = self.alloc_file_id(path);
        let change_kind = match (&self.get(file_id), &contents) {
            (None, None) => return,
            (None, Some(_)) => ChangeKind::Create,
            (Some(_), None) => ChangeKind::Delete,
            (Some(old), Some(new)) if old == new => return,
            (Some(_), Some(_)) => ChangeKind::Modify,
        };

        *self.get_mut(file_id) = contents;
        self.changes.push(ChangedFile { file_id, change_kind })
    }
    pub fn has_changes(&self) -> bool {
        !self.changes.is_empty()
    }
    pub fn take_changes(&mut self) -> Vec<ChangedFile> {
        mem::take(&mut self.changes)
    }
    fn alloc_file_id(&mut self, path: VfsPath) -> FileId {
        let file_id = self.interner.intern(path);
        let idx = file_id.0 as usize;
        let len = self.data.len().max(idx + 1);
        self.data.resize_with(len, || None);
        file_id
    }
    fn get(&self, file_id: FileId) -> &Option<Vec<u8>> {
        &self.data[file_id.0 as usize]
    }
    fn get_mut(&mut self, file_id: FileId) -> &mut Option<Vec<u8>> {
        &mut self.data[file_id.0 as usize]
    }
}

impl fmt::Debug for Vfs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vfs").field("n_files", &self.data.len()).finish()
    }
}
