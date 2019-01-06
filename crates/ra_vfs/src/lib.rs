//! VFS stands for Virtual File System.
//!
//! When doing analysis, we don't want to do any IO, we want to keep all source
//! code in memory. However, the actual source code is stored on disk, so you
//! need to get it into the memory in the first place somehow. VFS is the
//! component which does this.
//!
//! It is also responsible for watching the disk for changes, and for merging
//! editor state (modified, unsaved files) with disk state.
//! TODO: Some LSP clients support watching the disk, so this crate should
//! to support custom watcher events (related to https://github.com/rust-analyzer/rust-analyzer/issues/131)
//!
//! VFS is based on a concept of roots: a set of directories on the file system
//! which are watched for changes. Typically, there will be a root for each
//! Cargo package.
mod io;
mod watcher;

use std::{
    cmp::Reverse,
    ffi::OsStr,
    fmt, fs, mem,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

use crossbeam_channel::Receiver;
use ra_arena::{impl_arena_id, Arena, RawId};
use relative_path::RelativePathBuf;
use rustc_hash::{FxHashMap, FxHashSet};
use thread_worker::WorkerHandle;
use walkdir::DirEntry;

pub use crate::io::TaskResult as VfsTask;
pub use crate::watcher::{Watcher, WatcherChange};

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
        let path = path.strip_prefix(&self.root).ok()?;
        RelativePathBuf::from_path(path).ok()
    }
}

fn has_rs_extension(p: &Path) -> bool {
    p.extension() == Some(OsStr::new("rs"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VfsRoot(pub RawId);
impl_arena_id!(VfsRoot);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VfsFile(pub RawId);
impl_arena_id!(VfsFile);

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
    watcher: Watcher,
}

impl fmt::Debug for Vfs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Vfs { ... }")
    }
}

impl Vfs {
    pub fn new(mut roots: Vec<PathBuf>) -> (Vfs, Vec<VfsRoot>) {
        let (worker, worker_handle) = io::start();

        let watcher = Watcher::start().unwrap(); // TODO return Result?

        let mut res = Vfs {
            roots: Arena::default(),
            files: Arena::default(),
            root2files: FxHashMap::default(),
            worker,
            worker_handle,
            watcher,
            pending_changes: Vec::new(),
        };

        // A hack to make nesting work.
        roots.sort_by_key(|it| Reverse(it.as_os_str().len()));
        for (i, path) in roots.iter().enumerate() {
            let root = res.roots.alloc(RootFilter::new(path.clone()));
            res.root2files.insert(root, Default::default());
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
            res.worker.inp.send(task).unwrap();
            res.watcher.watch(path).unwrap();
        }
        let roots = res.roots.iter().map(|(id, _)| id).collect();
        (res, roots)
    }

    pub fn root2path(&self, root: VfsRoot) -> PathBuf {
        self.roots[root].root.clone()
    }

    pub fn path2file(&self, path: &Path) -> Option<VfsFile> {
        if let Some((_root, _path, Some(file))) = self.find_root(path) {
            return Some(file);
        }
        None
    }

    pub fn file2path(&self, file: VfsFile) -> PathBuf {
        let rel_path = &self.files[file].path;
        let root_path = &self.roots[self.files[file].root].root;
        rel_path.to_path(root_path)
    }

    pub fn file_for_path(&self, path: &Path) -> Option<VfsFile> {
        if let Some((_root, _path, Some(file))) = self.find_root(path) {
            return Some(file);
        }
        None
    }

    pub fn load(&mut self, path: &Path) -> Option<VfsFile> {
        if let Some((root, rel_path, file)) = self.find_root(path) {
            return if let Some(file) = file {
                Some(file)
            } else {
                let text = fs::read_to_string(path).unwrap_or_default();
                let text = Arc::new(text);
                let file = self.add_file(root, rel_path.clone(), Arc::clone(&text));
                let change = VfsChange::AddFile {
                    file,
                    text,
                    root,
                    path: rel_path,
                };
                self.pending_changes.push(change);
                Some(file)
            };
        }
        None
    }

    pub fn task_receiver(&self) -> &Receiver<io::TaskResult> {
        &self.worker.out
    }

    pub fn change_receiver(&self) -> &Receiver<WatcherChange> {
        &self.watcher.change_receiver()
    }

    pub fn handle_task(&mut self, task: io::TaskResult) {
        let mut files = Vec::new();
        // While we were scanning the root in the backgound, a file might have
        // been open in the editor, so we need to account for that.
        let exising = self.root2files[&task.root]
            .iter()
            .map(|&file| (self.files[file].path.clone(), file))
            .collect::<FxHashMap<_, _>>();
        for (path, text) in task.files {
            if let Some(&file) = exising.get(&path) {
                let text = Arc::clone(&self.files[file].text);
                files.push((file, path, text));
                continue;
            }
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

    pub fn handle_change(&mut self, change: WatcherChange) {
        match change {
            WatcherChange::Create(path) => {
                self.add_file_overlay(&path, None);
            }
            WatcherChange::Remove(path) => {
                self.remove_file_overlay(&path);
            }
            WatcherChange::Rename(src, dst) => {
                self.remove_file_overlay(&src);
                self.add_file_overlay(&dst, None);
            }
            WatcherChange::Write(path) => {
                self.change_file_overlay(&path, None);
            }
        }
    }

    pub fn add_file_overlay(&mut self, path: &Path, text: Option<String>) -> Option<VfsFile> {
        let mut res = None;
        if let Some((root, rel_path, file)) = self.find_root(path) {
            let text = text.unwrap_or_else(|| fs::read_to_string(&path).unwrap_or_default());
            let text = Arc::new(text);
            let change = if let Some(file) = file {
                res = Some(file);
                self.change_file(file, Arc::clone(&text));
                VfsChange::ChangeFile { file, text }
            } else {
                let file = self.add_file(root, rel_path.clone(), Arc::clone(&text));
                res = Some(file);
                VfsChange::AddFile {
                    file,
                    text,
                    root,
                    path: rel_path,
                }
            };
            self.pending_changes.push(change);
        }
        res
    }

    pub fn change_file_overlay(&mut self, path: &Path, new_text: Option<String>) {
        if let Some((_root, _path, file)) = self.find_root(path) {
            let new_text =
                new_text.unwrap_or_else(|| fs::read_to_string(&path).unwrap_or_default());
            let file = file.expect("can't change a file which wasn't added");
            let text = Arc::new(new_text);
            self.change_file(file, Arc::clone(&text));
            let change = VfsChange::ChangeFile { file, text };
            self.pending_changes.push(change);
        }
    }

    pub fn remove_file_overlay(&mut self, path: &Path) -> Option<VfsFile> {
        let mut res = None;
        if let Some((root, path, file)) = self.find_root(path) {
            let file = file.expect("can't remove a file which wasn't added");
            res = Some(file);
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
        res
    }

    pub fn commit_changes(&mut self) -> Vec<VfsChange> {
        mem::replace(&mut self.pending_changes, Vec::new())
    }

    /// Sutdown the VFS and terminate the background watching thread.
    pub fn shutdown(self) -> thread::Result<()> {
        let _ = self.watcher.shutdown();
        let _ = self.worker.shutdown();
        self.worker_handle.shutdown()
    }

    fn add_file(&mut self, root: VfsRoot, path: RelativePathBuf, text: Arc<String>) -> VfsFile {
        let data = VfsFileData { root, path, text };
        let file = self.files.alloc(data);
        self.root2files.get_mut(&root).unwrap().insert(file);
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
