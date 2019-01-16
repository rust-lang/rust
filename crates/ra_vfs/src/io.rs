use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use relative_path::RelativePathBuf;
use thread_worker::WorkerHandle;
use walkdir::{DirEntry, WalkDir};

use crate::{has_rs_extension, watcher::WatcherChange, VfsRoot};

pub(crate) enum Task {
    AddRoot {
        root: VfsRoot,
        path: PathBuf,
        filter: Box<Fn(&DirEntry) -> bool + Send>,
    },
    HandleChange(WatcherChange),
    LoadChange(WatcherChange),
}

#[derive(Debug)]
pub struct AddRootResult {
    pub(crate) root: VfsRoot,
    pub(crate) files: Vec<(RelativePathBuf, String)>,
}

#[derive(Debug)]
pub enum WatcherChangeData {
    Create { path: PathBuf, text: String },
    Write { path: PathBuf, text: String },
    Remove { path: PathBuf },
}

pub enum TaskResult {
    AddRoot(AddRootResult),
    HandleChange(WatcherChange),
    LoadChange(Option<WatcherChangeData>),
}

impl fmt::Debug for TaskResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("TaskResult { ... }")
    }
}

pub(crate) type Worker = thread_worker::Worker<Task, TaskResult>;

pub(crate) fn start() -> (Worker, WorkerHandle) {
    thread_worker::spawn("vfs", 128, |input_receiver, output_sender| {
        input_receiver
            .into_iter()
            .map(handle_task)
            .try_for_each(|it| output_sender.send(it))
            .unwrap()
    })
}

fn handle_task(task: Task) -> TaskResult {
    match task {
        Task::AddRoot { root, path, filter } => {
            log::debug!("loading {} ...", path.as_path().display());
            let files = load_root(path.as_path(), &*filter);
            log::debug!("... loaded {}", path.as_path().display());
            TaskResult::AddRoot(AddRootResult { root, files })
        }
        Task::HandleChange(change) => {
            // forward as is because Vfs has to decide if we should load it
            TaskResult::HandleChange(change)
        }
        Task::LoadChange(change) => {
            log::debug!("loading {:?} ...", change);
            let data = load_change(change);
            TaskResult::LoadChange(data)
        }
    }
}

fn load_root(root: &Path, filter: &dyn Fn(&DirEntry) -> bool) -> Vec<(RelativePathBuf, String)> {
    let mut res = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_entry(filter) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                log::warn!("watcher error: {}", e);
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !has_rs_extension(path) {
            continue;
        }
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => {
                log::warn!("watcher error: {}", e);
                continue;
            }
        };
        let path = RelativePathBuf::from_path(path.strip_prefix(root).unwrap()).unwrap();
        res.push((path.to_owned(), text))
    }
    res
}

fn load_change(change: WatcherChange) -> Option<WatcherChangeData> {
    let data = match change {
        WatcherChange::Create(path) => {
            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) => {
                    log::warn!("watcher error \"{}\": {}", path.display(), e);
                    return None;
                }
            };
            WatcherChangeData::Create { path, text }
        }
        WatcherChange::Write(path) => {
            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) => {
                    log::warn!("watcher error \"{}\": {}", path.display(), e);
                    return None;
                }
            };
            WatcherChangeData::Write { path, text }
        }
        WatcherChange::Remove(path) => WatcherChangeData::Remove { path },
        WatcherChange::Rescan => {
            // this should be handled by Vfs::handle_task
            return None;
        }
    };
    Some(data)
}
