//! VFS stands for Virtual File System.
//!
//! When doing analysis, we don't want to do any IO, we want to keep all source
//! code in memory. However, the actual source code is stored on disk, so you
//! need to get it into the memory in the first place somehow. VFS is the
//! component which does this.
//!
//! It also is responsible for watching the disk for changes, and for merging
//! editor state (modified, unsaved files) with disk state.
//!
//! VFS is based on a concept of roots: a set of directories on the file system
//! whihc are watched for changes. Typically, there will be a root for each
//! Cargo package.
mod arena;
mod io;

use std::{
    thread,
    cmp::Reverse,
    path::{Path, PathBuf},
    ffi::OsStr,
    sync::Arc,
};

use relative_path::RelativePathBuf;
use thread_worker::{WorkerHandle, Worker};

use crate::{
    arena::{ArenaId, Arena},
    io::FileEvent,
};

/// `RootFilter` is a predicate that checks if a file can belong to a root
struct RootFilter {
    root: PathBuf,
    file_filter: fn(&Path) -> bool,
}

impl RootFilter {
    fn new(root: PathBuf) -> RootFilter {
        RootFilter {
            root,
            file_filter: rs_extension_filter,
        }
    }
    fn can_contain(&self, path: &Path) -> bool {
        (self.file_filter)(path) && path.starts_with(&self.root)
    }
}

fn rs_extension_filter(p: &Path) -> bool {
    p.extension() == Some(OsStr::new("rs"))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct VfsRoot(u32);

impl ArenaId for VfsRoot {
    fn from_u32(idx: u32) -> VfsRoot {
        VfsRoot(idx)
    }
    fn to_u32(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct VfsFile(u32);

impl ArenaId for VfsFile {
    fn from_u32(idx: u32) -> VfsFile {
        VfsFile(idx)
    }
    fn to_u32(self) -> u32 {
        self.0
    }
}

struct VfsFileData {
    root: VfsRoot,
    path: RelativePathBuf,
    text: Arc<String>,
}

struct Vfs {
    roots: Arena<VfsRoot, RootFilter>,
    files: Arena<VfsFile, VfsFileData>,
    // pending_changes: Vec<PendingChange>,
    worker: Worker<PathBuf, (PathBuf, Vec<FileEvent>)>,
    worker_handle: WorkerHandle,
}

impl Vfs {
    pub fn new(mut roots: Vec<PathBuf>) -> Vfs {
        let (worker, worker_handle) = io::start();

        let mut res = Vfs {
            roots: Arena::default(),
            files: Arena::default(),
            worker,
            worker_handle,
        };

        roots.sort_by_key(|it| Reverse(it.as_os_str().len()));

        for path in roots {
            res.roots.alloc(RootFilter::new(path));
        }
        res
    }

    pub fn add_file_overlay(&mut self, path: &Path, content: String) {}

    pub fn change_file_overlay(&mut self, path: &Path, new_content: String) {}

    pub fn remove_file_overlay(&mut self, path: &Path) {}

    pub fn commit_changes(&mut self) -> Vec<VfsChange> {
        unimplemented!()
    }

    pub fn shutdown(self) -> thread::Result<()> {
        let _ = self.worker.shutdown();
        self.worker_handle.shutdown()
    }
}

#[derive(Debug, Clone)]
pub enum VfsChange {
    AddRoot {
        root: VfsRoot,
        files: Vec<(VfsFile, RelativePathBuf, Arc<String>)>,
    },
    AddFile {
        file: VfsFile,
        root: VfsRoot,
        path: RelativePathBuf,
        text: Arc<String>,
    },
    RemoveFile {
        file: VfsFile,
    },
    ChangeFile {
        file: VfsFile,
        text: Arc<String>,
    },
}
