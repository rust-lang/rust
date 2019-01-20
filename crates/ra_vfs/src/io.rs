use std::{
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use relative_path::RelativePathBuf;
use thread_worker::WorkerHandle;
use walkdir::{DirEntry, WalkDir};

use crate::{
    watcher::{Watcher, WatcherChange},
    VfsRoot,
};

pub(crate) enum Task {
    AddRoot {
        root: VfsRoot,
        path: PathBuf,
        filter: Box<Fn(&DirEntry) -> bool + Send>,
    },
    HandleChange(WatcherChange),
    LoadChange(WatcherChange),
    Watch {
        dir: PathBuf,
        filter: Box<Fn(&DirEntry) -> bool + Send>,
    },
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
    LoadChange(WatcherChangeData),
    NoOp,
}

impl fmt::Debug for TaskResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("TaskResult { ... }")
    }
}

pub(crate) struct Worker {
    worker: thread_worker::Worker<Task, TaskResult>,
    worker_handle: WorkerHandle,
    watcher: Arc<Mutex<Option<Watcher>>>,
}

impl Worker {
    pub(crate) fn start() -> Worker {
        let watcher = Arc::new(Mutex::new(None));
        let watcher_clone = watcher.clone();
        let (worker, worker_handle) =
            thread_worker::spawn("vfs", 128, move |input_receiver, output_sender| {
                let res = input_receiver
                    .into_iter()
                    .map(|t| handle_task(t, &watcher_clone))
                    .try_for_each(|it| output_sender.send(it));
                res.unwrap()
            });
        match Watcher::start(worker.inp.clone()) {
            Ok(w) => {
                watcher.lock().replace(w);
            }
            Err(e) => log::error!("could not start watcher: {}", e),
        };
        Worker {
            worker,
            worker_handle,
            watcher,
        }
    }

    pub(crate) fn sender(&self) -> &Sender<Task> {
        &self.worker.inp
    }

    pub(crate) fn receiver(&self) -> &Receiver<TaskResult> {
        &self.worker.out
    }

    pub(crate) fn shutdown(self) -> thread::Result<()> {
        if let Some(watcher) = self.watcher.lock().take() {
            let _ = watcher.shutdown();
        }
        self.worker_handle.shutdown()
    }
}

fn watch(
    watcher: &Arc<Mutex<Option<Watcher>>>,
    dir: &Path,
    filter_entry: impl Fn(&DirEntry) -> bool,
    emit_for_existing: bool,
) {
    let mut watcher = watcher.lock();
    let watcher = match *watcher {
        Some(ref mut w) => w,
        None => {
            // watcher dropped or couldn't start
            return;
        }
    };
    watcher.watch_recursive(dir, filter_entry, emit_for_existing)
}

fn handle_task(task: Task, watcher: &Arc<Mutex<Option<Watcher>>>) -> TaskResult {
    match task {
        Task::AddRoot { root, path, filter } => {
            watch(watcher, &path, &*filter, false);
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
            match load_change(change) {
                Some(data) => TaskResult::LoadChange(data),
                None => TaskResult::NoOp,
            }
        }
        Task::Watch { dir, filter } => {
            watch(watcher, &dir, &*filter, true);
            TaskResult::NoOp
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
            if path.is_dir() {
                return None;
            }
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
