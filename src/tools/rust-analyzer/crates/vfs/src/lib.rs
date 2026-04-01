//! # Virtual File System
//!
//! VFS records all file changes pushed to it via [`set_file_contents`].
//! As such it only ever stores changes, not the actual content of a file at any given moment.
//! All file changes are logged, and can be retrieved via
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

mod anchored_path;
pub mod file_set;
pub mod loader;
mod path_interner;
mod vfs_path;

use std::{fmt, hash::BuildHasherDefault, mem};

use crate::path_interner::PathInterner;

pub use crate::{
    anchored_path::{AnchoredPath, AnchoredPathBuf},
    vfs_path::VfsPath,
};
use indexmap::{IndexMap, map::Entry};
pub use paths::{AbsPath, AbsPathBuf};

use rustc_hash::FxHasher;
use stdx::hash_once;
use tracing::{Level, span};

/// Handle to a file in [`Vfs`]
///
/// Most functions in rust-analyzer use this when they need to refer to a file.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct FileId(u32);
// pub struct FileId(NonMaxU32);

impl FileId {
    const MAX: u32 = 0x7fff_ffff;

    #[inline]
    pub const fn from_raw(raw: u32) -> FileId {
        assert!(raw <= Self::MAX);
        FileId(raw)
    }

    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// safe because `FileId` is a newtype of `u32`
impl nohash_hasher::IsEnabled for FileId {}

/// Storage for all file changes and the file id to path mapping.
///
/// For more information see the [crate-level](crate) documentation.
#[derive(Default)]
pub struct Vfs {
    interner: PathInterner,
    data: Vec<FileState>,
    changes: IndexMap<FileId, ChangedFile, BuildHasherDefault<FxHasher>>,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FileState {
    /// The file exists with the given content hash.
    Exists(u64),
    /// The file is deleted.
    Deleted,
    /// The file was specifically excluded by the user. We still include excluded files
    /// when they're opened (without their contents).
    Excluded,
}

/// Changed file in the [`Vfs`].
#[derive(Debug)]
pub struct ChangedFile {
    /// Id of the changed file
    pub file_id: FileId,
    /// Kind of change
    pub change: Change,
}

impl ChangedFile {
    /// Returns `true` if the change is not [`Delete`](ChangeKind::Delete).
    pub fn exists(&self) -> bool {
        !matches!(self.change, Change::Delete)
    }

    /// Returns `true` if the change is [`Create`](ChangeKind::Create) or
    /// [`Delete`](Change::Delete).
    pub fn is_created_or_deleted(&self) -> bool {
        matches!(self.change, Change::Create(_, _) | Change::Delete)
    }

    /// Returns `true` if the change is [`Create`](ChangeKind::Create).
    pub fn is_created(&self) -> bool {
        matches!(self.change, Change::Create(_, _))
    }

    /// Returns `true` if the change is [`Modify`](ChangeKind::Modify).
    pub fn is_modified(&self) -> bool {
        matches!(self.change, Change::Modify(_, _))
    }

    pub fn kind(&self) -> ChangeKind {
        match self.change {
            Change::Create(_, _) => ChangeKind::Create,
            Change::Modify(_, _) => ChangeKind::Modify,
            Change::Delete => ChangeKind::Delete,
        }
    }
}

/// Kind of [file change](ChangedFile).
#[derive(Eq, PartialEq, Debug)]
pub enum Change {
    /// The file was (re-)created
    Create(Vec<u8>, u64),
    /// The file was modified
    Modify(Vec<u8>, u64),
    /// The file was deleted
    Delete,
}

/// Kind of [file change](ChangedFile).
#[derive(Eq, PartialEq, Debug)]
pub enum ChangeKind {
    /// The file was (re-)created
    Create,
    /// The file was modified
    Modify,
    /// The file was deleted
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileExcluded {
    Yes,
    No,
}

impl Vfs {
    /// Id of the given path if it exists in the `Vfs` and is not deleted.
    pub fn file_id(&self, path: &VfsPath) -> Option<(FileId, FileExcluded)> {
        let file_id = self.interner.get(path)?;
        let file_state = self.get(file_id);
        match file_state {
            FileState::Exists(_) => Some((file_id, FileExcluded::No)),
            FileState::Deleted => None,
            FileState::Excluded => Some((file_id, FileExcluded::Yes)),
        }
    }

    /// File path corresponding to the given `file_id`.
    ///
    /// # Panics
    ///
    /// Panics if the id is not present in the `Vfs`.
    pub fn file_path(&self, file_id: FileId) -> &VfsPath {
        self.interner.lookup(file_id)
    }

    /// Returns an iterator over the stored ids and their corresponding paths.
    ///
    /// This will skip deleted files.
    pub fn iter(&self) -> impl Iterator<Item = (FileId, &VfsPath)> + '_ {
        (0..self.data.len())
            .map(|it| FileId(it as u32))
            .filter(move |&file_id| matches!(self.get(file_id), FileState::Exists(_)))
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
        let _p = span!(Level::INFO, "Vfs::set_file_contents").entered();
        let file_id = self.alloc_file_id(path);
        let state: FileState = self.get(file_id);
        let change = match (state, contents) {
            (FileState::Deleted, None) => return false,
            (FileState::Deleted, Some(v)) => {
                let hash = hash_once::<FxHasher>(&*v);
                Change::Create(v, hash)
            }
            (FileState::Exists(_), None) => Change::Delete,
            (FileState::Exists(hash), Some(v)) => {
                let new_hash = hash_once::<FxHasher>(&*v);
                if new_hash == hash {
                    return false;
                }
                Change::Modify(v, new_hash)
            }
            (FileState::Excluded, _) => return false,
        };

        let mut set_data = |change_kind| {
            self.data[file_id.0 as usize] = match change_kind {
                &Change::Create(_, hash) | &Change::Modify(_, hash) => FileState::Exists(hash),
                Change::Delete => FileState::Deleted,
            };
        };

        let changed_file = ChangedFile { file_id, change };
        match self.changes.entry(file_id) {
            // two changes to the same file in one cycle, merge them appropriately
            Entry::Occupied(mut o) => {
                use Change::*;

                match (&mut o.get_mut().change, changed_file.change) {
                    // newer `Delete` wins
                    (change, Delete) => *change = Delete,
                    // merge `Create` with `Create` or `Modify`
                    (Create(prev, old_hash), Create(new, new_hash) | Modify(new, new_hash)) => {
                        *prev = new;
                        *old_hash = new_hash;
                    }
                    // collapse identical `Modify`es
                    (Modify(prev, old_hash), Modify(new, new_hash)) => {
                        *prev = new;
                        *old_hash = new_hash;
                    }
                    // equivalent to `Modify`
                    (change @ Delete, Create(new, new_hash)) => {
                        *change = Modify(new, new_hash);
                    }
                    // shouldn't occur, but collapse into `Create`
                    (change @ Delete, Modify(new, new_hash)) => {
                        stdx::never!();
                        *change = Create(new, new_hash);
                    }
                    // shouldn't occur, but keep the Create
                    (prev @ Modify(_, _), new @ Create(_, _)) => *prev = new,
                }
                set_data(&o.get().change);
            }
            Entry::Vacant(v) => set_data(&v.insert(changed_file).change),
        };

        true
    }

    /// Drain and returns all the changes in the `Vfs`.
    pub fn take_changes(&mut self) -> IndexMap<FileId, ChangedFile, BuildHasherDefault<FxHasher>> {
        mem::take(&mut self.changes)
    }

    /// Provides a panic-less way to verify file_id validity.
    pub fn exists(&self, file_id: FileId) -> bool {
        matches!(self.get(file_id), FileState::Exists(_))
    }

    /// Returns the id associated with `path`
    ///
    /// - If `path` does not exists in the `Vfs`, allocate a new id for it, associated with a
    ///   deleted file;
    /// - Else, returns `path`'s id.
    ///
    /// Does not record a change.
    fn alloc_file_id(&mut self, path: VfsPath) -> FileId {
        let file_id = self.interner.intern(path);
        let idx = file_id.0 as usize;
        let len = self.data.len().max(idx + 1);
        self.data.resize(len, FileState::Deleted);
        file_id
    }

    /// Returns the status of the file associated with the given `file_id`.
    ///
    /// # Panics
    ///
    /// Panics if no file is associated to that id.
    fn get(&self, file_id: FileId) -> FileState {
        self.data[file_id.0 as usize]
    }

    /// We cannot ignore excluded files, because this will lead to errors when the client
    /// requests semantic information for them, so we instead mark them specially.
    pub fn insert_excluded_file(&mut self, path: VfsPath) {
        let file_id = self.alloc_file_id(path);
        self.data[file_id.0 as usize] = FileState::Excluded;
    }
}

impl fmt::Debug for Vfs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vfs").field("n_files", &self.data.len()).finish()
    }
}
