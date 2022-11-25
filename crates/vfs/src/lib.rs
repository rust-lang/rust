//! # Virtual File System
//!
//! VFS stores all files read by rust-analyzer. Reading file contents from VFS
//! always returns the same contents, unless VFS was explicitly modified with
//! [`set_file_contents`]. All changes to VFS are logged, and can be retrieved via
//! [`take_changes`] method. The pack of changes is then pushed to `salsa` and
//! triggers incremental recomputation.
//!
//! Files in VFS are identified with [`FileId`]s -- interned paths. The notion of
//! the path, [`VfsPath`] is somewhat abstract: at the moment, it is represented
//! as an [`std::path::PathBuf`] internally, but this is an implementation detail.
//!
//! VFS doesn't do IO or file watching itself. For that, see the [`loader`]
//! module. [`loader::Handle`] is an object-safe trait which abstracts both file
//! loading and file watching. [`Handle`] is dynamically configured with a set of
//! directory entries which should be scanned and watched. [`Handle`] then
//! asynchronously pushes file changes. Directory entries are configured in
//! free-form via list of globs, it's up to the [`Handle`] to interpret the globs
//! in any specific way.
//!
//! VFS stores a flat list of files. [`file_set::FileSet`] can partition this list
//! of files into disjoint sets of files. Traversal-like operations (including
//! getting the neighbor file by the relative path) are handled by the [`FileSet`].
//! [`FileSet`]s are also pushed to salsa and cause it to re-check `mod foo;`
//! declarations when files are created or deleted.
//!
//! [`FileSet`] and [`loader::Entry`] play similar, but different roles.
//! Both specify the "set of paths/files", one is geared towards file watching,
//! the other towards salsa changes. In particular, single [`FileSet`]
//! may correspond to several [`loader::Entry`]. For example, a crate from
//! crates.io which uses code generation would have two [`Entries`] -- for sources
//! in `~/.cargo`, and for generated code in `./target/debug/build`. It will
//! have a single [`FileSet`] which unions the two sources.
//!
//! [`set_file_contents`]: Vfs::set_file_contents
//! [`take_changes`]: Vfs::take_changes
//! [`FileSet`]: file_set::FileSet
//! [`Handle`]: loader::Handle
//! [`Entries`]: loader::Entry

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod anchored_path;
pub mod file_set;
pub mod loader;
mod path_interner;
mod vfs_path;

use std::{fmt, mem};

use crate::path_interner::PathInterner;

pub use crate::{
    anchored_path::{AnchoredPath, AnchoredPathBuf},
    vfs_path::VfsPath,
};
pub use paths::{AbsPath, AbsPathBuf};

/// Handle to a file in [`Vfs`]
///
/// Most functions in rust-analyzer use this when they need to refer to a file.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct FileId(pub u32);

/// Storage for all files read by rust-analyzer.
///
/// For more information see the [crate-level](crate) documentation.
#[derive(Default)]
pub struct Vfs {
    interner: PathInterner,
    data: Vec<Option<Vec<u8>>>,
    changes: Vec<ChangedFile>,
}

/// Changed file in the [`Vfs`].
pub struct ChangedFile {
    /// Id of the changed file
    pub file_id: FileId,
    /// Kind of change
    pub change_kind: ChangeKind,
}

impl ChangedFile {
    /// Returns `true` if the change is not [`Delete`](ChangeKind::Delete).
    pub fn exists(&self) -> bool {
        self.change_kind != ChangeKind::Delete
    }

    /// Returns `true` if the change is [`Create`](ChangeKind::Create) or
    /// [`Delete`](ChangeKind::Delete).
    pub fn is_created_or_deleted(&self) -> bool {
        matches!(self.change_kind, ChangeKind::Create | ChangeKind::Delete)
    }
}

/// Kind of [file change](ChangedFile).
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum ChangeKind {
    /// The file was (re-)created
    Create,
    /// The file was modified
    Modify,
    /// The file was deleted
    Delete,
}

impl Vfs {
    /// Amount of files currently stored.
    ///
    /// Note that this includes deleted files.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Id of the given path if it exists in the `Vfs` and is not deleted.
    pub fn file_id(&self, path: &VfsPath) -> Option<FileId> {
        self.interner.get(path).filter(|&it| self.get(it).is_some())
    }

    /// File path corresponding to the given `file_id`.
    ///
    /// # Panics
    ///
    /// Panics if the id is not present in the `Vfs`.
    pub fn file_path(&self, file_id: FileId) -> VfsPath {
        self.interner.lookup(file_id).clone()
    }

    /// File content corresponding to the given `file_id`.
    ///
    /// # Panics
    ///
    /// Panics if the id is not present in the `Vfs`, or if the corresponding file is
    /// deleted.
    pub fn file_contents(&self, file_id: FileId) -> &[u8] {
        self.get(file_id).as_deref().unwrap()
    }

    /// Returns an iterator over the stored ids and their corresponding paths.
    ///
    /// This will skip deleted files.
    pub fn iter(&self) -> impl Iterator<Item = (FileId, &VfsPath)> + '_ {
        (0..self.data.len())
            .map(|it| FileId(it as u32))
            .filter(move |&file_id| self.get(file_id).is_some())
            .map(move |file_id| {
                let path = self.interner.lookup(file_id);
                (file_id, path)
            })
    }

    /// Update the `path` with the given `contents`. `None` means the file was deleted.
    ///
    /// Returns `true` if the file was modified, and saves the [change](ChangedFile).
    ///
    /// If the path does not currently exists in the `Vfs`, allocates a new
    /// [`FileId`] for it.
    pub fn set_file_contents(&mut self, path: VfsPath, contents: Option<Vec<u8>>) -> bool {
        let file_id = self.alloc_file_id(path);
        let change_kind = match (&self.get(file_id), &contents) {
            (None, None) => return false,
            (None, Some(_)) => ChangeKind::Create,
            (Some(_), None) => ChangeKind::Delete,
            (Some(old), Some(new)) if old == new => return false,
            (Some(_), Some(_)) => ChangeKind::Modify,
        };

        *self.get_mut(file_id) = contents;
        self.changes.push(ChangedFile { file_id, change_kind });
        true
    }

    /// Returns `true` if the `Vfs` contains [changes](ChangedFile).
    pub fn has_changes(&self) -> bool {
        !self.changes.is_empty()
    }

    /// Drain and returns all the changes in the `Vfs`.
    pub fn take_changes(&mut self) -> Vec<ChangedFile> {
        mem::take(&mut self.changes)
    }

    /// Returns the id associated with `path`
    ///
    /// - If `path` does not exists in the `Vfs`, allocate a new id for it, associated with a
    /// deleted file;
    /// - Else, returns `path`'s id.
    ///
    /// Does not record a change.
    fn alloc_file_id(&mut self, path: VfsPath) -> FileId {
        let file_id = self.interner.intern(path);
        let idx = file_id.0 as usize;
        let len = self.data.len().max(idx + 1);
        self.data.resize_with(len, || None);
        file_id
    }

    /// Returns the content associated with the given `file_id`.
    ///
    /// # Panics
    ///
    /// Panics if no file is associated to that id.
    fn get(&self, file_id: FileId) -> &Option<Vec<u8>> {
        &self.data[file_id.0 as usize]
    }

    /// Mutably returns the content associated with the given `file_id`.
    ///
    /// # Panics
    ///
    /// Panics if no file is associated to that id.
    fn get_mut(&mut self, file_id: FileId) -> &mut Option<Vec<u8>> {
        &mut self.data[file_id.0 as usize]
    }
}

impl fmt::Debug for Vfs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vfs").field("n_files", &self.data.len()).finish()
    }
}
