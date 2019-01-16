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

pub(crate) fn has_rs_extension(p: &Path) -> bool {
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
    is_overlayed: bool,
    text: Arc<String>,
}

pub struct Vfs {
    roots: Arena<VfsRoot, RootFilter>,
    files: Arena<VfsFile, VfsFileData>,
    root2files: FxHashMap<VfsRoot, FxHashSet<VfsFile>>,
    pending_changes: Vec<VfsChange>,
    worker: io::Worker,
    worker_handle: WorkerHandle,
    watcher: Option<Watcher>,
}

impl fmt::Debug for Vfs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Vfs { ... }")
    }
}

impl Vfs {
    pub fn new(mut roots: Vec<PathBuf>) -> (Vfs, Vec<VfsRoot>) {
        let (worker, worker_handle) = io::start();

        let watcher = match Watcher::start(worker.inp.clone()) {
            Ok(watcher) => Some(watcher),
            Err(e) => {
                log::error!("could not start watcher: {}", e);
                None
            }
        };

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
            let task = io::Task::AddRoot {
                root,
                path: path.clone(),
                filter: Box::new(filter),
            };
            res.worker.inp.send(task).unwrap();
            if let Some(ref mut watcher) = res.watcher {
                if let Err(e) = watcher.watch(path) {
                    log::warn!("could not watch \"{}\": {}", path.display(), e);
                }
            }
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
                let file = self.add_file(root, rel_path.clone(), Arc::clone(&text), false);
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

    pub fn handle_task(&mut self, task: io::TaskResult) {
        match task {
            io::TaskResult::AddRoot(task) => {
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
                    let file = self.add_file(task.root, path.clone(), Arc::clone(&text), false);
                    files.push((file, path, text));
                }

                let change = VfsChange::AddRoot {
                    root: task.root,
                    files,
                };
                self.pending_changes.push(change);
            }
            io::TaskResult::HandleChange(change) => match &change {
                watcher::WatcherChange::Create(path)
                | watcher::WatcherChange::Remove(path)
                | watcher::WatcherChange::Write(path) => {
                    if self.should_handle_change(&path) {
                        self.worker.inp.send(io::Task::LoadChange(change)).unwrap()
                    }
                }
                watcher::WatcherChange::Rescan => {
                    // TODO we should reload all files
                }
            },
            io::TaskResult::LoadChange(None) => {}
            io::TaskResult::LoadChange(Some(change)) => match change {
                io::WatcherChangeData::Create { path, text }
                | io::WatcherChangeData::Write { path, text } => {
                    if let Some((root, path, file)) = self.find_root(&path) {
                        if let Some(file) = file {
                            self.do_change_file(file, text, false);
                        } else {
                            self.do_add_file(root, path, text, false);
                        }
                    }
                }
                io::WatcherChangeData::Remove { path } => {
                    if let Some((root, path, file)) = self.find_root(&path) {
                        if let Some(file) = file {
                            self.do_remove_file(root, path, file, false);
                        }
                    }
                }
            },
        }
    }

    fn should_handle_change(&self, path: &Path) -> bool {
        if let Some((_root, _rel_path, file)) = self.find_root(&path) {
            if let Some(file) = file {
                if self.files[file].is_overlayed {
                    // file is overlayed
                    log::debug!("skipping overlayed \"{}\"", path.display());
                    return false;
                }
            }
            true
        } else {
            // file doesn't belong to any root
            false
        }
    }

    fn do_add_file(
        &mut self,
        root: VfsRoot,
        path: RelativePathBuf,
        text: String,
        is_overlay: bool,
    ) -> Option<VfsFile> {
        let text = Arc::new(text);
        let file = self.add_file(root, path.clone(), text.clone(), is_overlay);
        self.pending_changes.push(VfsChange::AddFile {
            file,
            root,
            path,
            text,
        });
        Some(file)
    }

    fn do_change_file(&mut self, file: VfsFile, text: String, is_overlay: bool) {
        if !is_overlay && self.files[file].is_overlayed {
            return;
        }
        let text = Arc::new(text);
        self.change_file(file, text.clone(), is_overlay);
        self.pending_changes
            .push(VfsChange::ChangeFile { file, text });
    }

    fn do_remove_file(
        &mut self,
        root: VfsRoot,
        path: RelativePathBuf,
        file: VfsFile,
        is_overlay: bool,
    ) {
        if !is_overlay && self.files[file].is_overlayed {
            return;
        }
        self.remove_file(file);
        self.pending_changes
            .push(VfsChange::RemoveFile { root, path, file });
    }

    pub fn add_file_overlay(&mut self, path: &Path, text: String) -> Option<VfsFile> {
        if let Some((root, rel_path, file)) = self.find_root(path) {
            if let Some(file) = file {
                self.do_change_file(file, text, true);
                Some(file)
            } else {
                self.do_add_file(root, rel_path, text, true)
            }
        } else {
            None
        }
    }

    pub fn change_file_overlay(&mut self, path: &Path, new_text: String) {
        if let Some((_root, _path, file)) = self.find_root(path) {
            let file = file.expect("can't change a file which wasn't added");
            self.do_change_file(file, new_text, true);
        }
    }

    pub fn remove_file_overlay(&mut self, path: &Path) -> Option<VfsFile> {
        if let Some((root, path, file)) = self.find_root(path) {
            let file = file.expect("can't remove a file which wasn't added");
            let full_path = path.to_path(&self.roots[root].root);
            if let Ok(text) = fs::read_to_string(&full_path) {
                self.do_change_file(file, text, true);
            } else {
                self.do_remove_file(root, path, file, true);
            }
            Some(file)
        } else {
            None
        }
    }

    pub fn commit_changes(&mut self) -> Vec<VfsChange> {
        mem::replace(&mut self.pending_changes, Vec::new())
    }

    /// Sutdown the VFS and terminate the background watching thread.
    pub fn shutdown(self) -> thread::Result<()> {
        if let Some(watcher) = self.watcher {
            let _ = watcher.shutdown();
        }
        let _ = self.worker.shutdown();
        self.worker_handle.shutdown()
    }

    fn add_file(
        &mut self,
        root: VfsRoot,
        path: RelativePathBuf,
        text: Arc<String>,
        is_overlayed: bool,
    ) -> VfsFile {
        let data = VfsFileData {
            root,
            path,
            text,
            is_overlayed,
        };
        let file = self.files.alloc(data);
        self.root2files.get_mut(&root).unwrap().insert(file);
        file
    }

    fn change_file(&mut self, file: VfsFile, new_text: Arc<String>, is_overlayed: bool) {
        let mut file_data = &mut self.files[file];
        file_data.text = new_text;
        file_data.is_overlayed = is_overlayed;
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
