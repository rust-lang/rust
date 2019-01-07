use std::{
    fmt,
    fs,
    path::{Path, PathBuf},
};

use walkdir::{DirEntry, WalkDir};
use thread_worker::{WorkerHandle};
use relative_path::RelativePathBuf;

use crate::{VfsRoot, has_rs_extension};

pub(crate) enum Task {
    AddRoot {
        root: VfsRoot,
        path: PathBuf,
        filter: Box<Fn(&DirEntry) -> bool + Send>,
    },
    WatcherChange(crate::watcher::WatcherChange),
}

#[derive(Debug)]
pub struct AddRootResult {
    pub(crate) root: VfsRoot,
    pub(crate) files: Vec<(RelativePathBuf, String)>,
}

#[derive(Debug)]
pub enum WatcherChangeResult {
    Create {
        path: PathBuf,
        text: String,
    },
    Write {
        path: PathBuf,
        text: String,
    },
    Remove {
        path: PathBuf,
    },
    // can this be replaced and use Remove and Create instead?
    Rename {
        src: PathBuf,
        dst: PathBuf,
        text: String,
    },
}

pub enum TaskResult {
    AddRoot(AddRootResult),
    WatcherChange(WatcherChangeResult),
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
        Task::WatcherChange(change) => {
            // TODO
            unimplemented!()
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
