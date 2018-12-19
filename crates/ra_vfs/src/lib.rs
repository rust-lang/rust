//! VFS stands for Virtual File System.
//!
//! When doing analysis, we don't want to do any IO, we want to keep all source
//! code in memory. However, the actual source code is stored on disk, so you
//! component which does this.
//! need to get it into the memory in the first place somehow. VFS is the
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
    mem,
    thread,
    cmp::Reverse,
    path::{Path, PathBuf},
    ffi::OsStr,
    sync::Arc,
    fs,
};

use rustc_hash::{FxHashMap, FxHashSet};
use relative_path::RelativePathBuf;
use crossbeam_channel::Receiver;
use walkdir::DirEntry;
use thread_worker::{WorkerHandle};

use crate::{
    arena::{ArenaId, Arena},
};

/// `RootFilter` is a predicate that checks if a file can belong to a root. If
/// several filters match a file (nested dirs), the most nested one wins.
struct RootFilter {
    root: PathBuf,
    file_filter: fn(&Path) -> bool,
}

impl RootFilter {
    fn new(root: PathBuf) -> RootFilter {
        RootFilter {
            root,
            file_filter: has_rs_extension,
        }
    }
    /// Check if this root can contain `path`. NB: even if this returns
    /// true, the `path` might actually be conained in some nested root.
    fn can_contain(&self, path: &Path) -> Option<RelativePathBuf> {
        if !(self.file_filter)(path) {
            return None;
        }
        if !(path.starts_with(&self.root)) {
            return None;
        }
        let path = path.strip_prefix(&self.root).unwrap();
        let path = RelativePathBuf::from_path(path).unwrap();
        Some(path)
    }
}

fn has_rs_extension(p: &Path) -> bool {
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

pub struct Vfs {
    roots: Arena<VfsRoot, RootFilter>,
    files: Arena<VfsFile, VfsFileData>,
    root2files: FxHashMap<VfsRoot, FxHashSet<VfsFile>>,
    pending_changes: Vec<VfsChange>,
    worker: io::Worker,
    worker_handle: WorkerHandle,
}

impl Vfs {
    pub fn new(mut roots: Vec<PathBuf>) -> Vfs {
        let (worker, worker_handle) = io::start();

        let mut res = Vfs {
            roots: Arena::default(),
            files: Arena::default(),
            root2files: FxHashMap::default(),
            worker,
            worker_handle,
            pending_changes: Vec::new(),
        };

        // A hack to make nesting work.
        roots.sort_by_key(|it| Reverse(it.as_os_str().len()));
        for (i, path) in roots.iter().enumerate() {
            let root = res.roots.alloc(RootFilter::new(path.clone()));
            let nested = roots[..i]
                .iter()
                .filter(|it| it.starts_with(path))
                .map(|it| it.clone())
                .collect::<Vec<_>>();
            let filter = move |entry: &DirEntry| {
                if entry.file_type().is_file() {
                    has_rs_extension(entry.path())
                } else {
                    nested.iter().all(|it| it != entry.path())
                }
            };
            let task = io::Task {
                root,
                path: path.clone(),
                filter: Box::new(filter),
            };
            res.worker.inp.send(task);
        }
        res
    }

    pub fn task_receiver(&self) -> &Receiver<io::TaskResult> {
        &self.worker.out
    }

    pub fn handle_task(&mut self, task: io::TaskResult) {
        let mut files = Vec::new();
        for (path, text) in task.files {
            let text = Arc::new(text);
            let file = self.add_file(task.root, path.clone(), Arc::clone(&text));
            files.push((file, path, text));
        }
        let change = VfsChange::AddRoot {
            root: task.root,
            files,
        };
        self.pending_changes.push(change);
    }

    pub fn add_file_overlay(&mut self, path: &Path, text: String) {
        if let Some((root, path, file)) = self.find_root(path) {
            let text = Arc::new(text);
            let change = if let Some(file) = file {
                self.change_file(file, Arc::clone(&text));
                VfsChange::ChangeFile { file, text }
            } else {
                let file = self.add_file(root, path.clone(), Arc::clone(&text));
                VfsChange::AddFile {
                    file,
                    text,
                    root,
                    path,
                }
            };
            self.pending_changes.push(change);
        }
    }

    pub fn change_file_overlay(&mut self, path: &Path, new_text: String) {
        if let Some((_root, _path, file)) = self.find_root(path) {
            let file = file.expect("can't change a file which wasn't added");
            let text = Arc::new(new_text);
            self.change_file(file, Arc::clone(&text));
            let change = VfsChange::ChangeFile { file, text };
            self.pending_changes.push(change);
        }
    }

    pub fn remove_file_overlay(&mut self, path: &Path) {
        if let Some((root, path, file)) = self.find_root(path) {
            let file = file.expect("can't remove a file which wasn't added");
            let full_path = path.to_path(&self.roots[root].root);
            let change = if let Ok(text) = fs::read_to_string(&full_path) {
                let text = Arc::new(text);
                self.change_file(file, Arc::clone(&text));
                VfsChange::ChangeFile { file, text }
            } else {
                self.remove_file(file);
                VfsChange::RemoveFile { root, file, path }
            };
            self.pending_changes.push(change);
        }
    }

    pub fn commit_changes(&mut self) -> Vec<VfsChange> {
        mem::replace(&mut self.pending_changes, Vec::new())
    }

    pub fn shutdown(self) -> thread::Result<()> {
        let _ = self.worker.shutdown();
        self.worker_handle.shutdown()
    }

    fn add_file(&mut self, root: VfsRoot, path: RelativePathBuf, text: Arc<String>) -> VfsFile {
        let data = VfsFileData { root, path, text };
        let file = self.files.alloc(data);
        self.root2files
            .entry(root)
            .or_insert_with(FxHashSet::default)
            .insert(file);
        file
    }

    fn change_file(&mut self, file: VfsFile, new_text: Arc<String>) {
        self.files[file].text = new_text;
    }

    fn remove_file(&mut self, file: VfsFile) {
        //FIXME: use arena with removal
        self.files[file].text = Default::default();
        self.files[file].path = Default::default();
        let root = self.files[file].root;
        let removed = self.root2files.get_mut(&root).unwrap().remove(&file);
        assert!(removed);
    }

    fn find_root(&self, path: &Path) -> Option<(VfsRoot, RelativePathBuf, Option<VfsFile>)> {
        let (root, path) = self
            .roots
            .iter()
            .find_map(|(root, data)| data.can_contain(path).map(|it| (root, it)))?;
        let file = self.root2files[&root]
            .iter()
            .map(|&it| it)
            .find(|&file| self.files[file].path == path);
        Some((root, path, file))
    }
}

#[derive(Debug, Clone)]
pub enum VfsChange {
    AddRoot {
        root: VfsRoot,
        files: Vec<(VfsFile, RelativePathBuf, Arc<String>)>,
    },
    AddFile {
        root: VfsRoot,
        file: VfsFile,
        path: RelativePathBuf,
        text: Arc<String>,
    },
    RemoveFile {
        root: VfsRoot,
        file: VfsFile,
        path: RelativePathBuf,
    },
    ChangeFile {
        file: VfsFile,
        text: Arc<String>,
    },
}
