use std::{
    fs,
    path::{Path, PathBuf},
    thread::JoinHandle,
};

use walkdir::{DirEntry, WalkDir};
use crossbeam_channel::{Sender, Receiver};
use thread_worker::{WorkerHandle};

use crate::VfsRoot;

pub(crate) enum Task {
    ScanRoot {
        root: VfsRoot,
        path: PathBuf,
        filter: Box<FnMut(&DirEntry) -> bool + Send>,
    },
}

#[derive(Debug)]
pub(crate) struct FileEvent {
    pub(crate) path: PathBuf,
    pub(crate) kind: FileEventKind,
}

#[derive(Debug)]
pub(crate) enum FileEventKind {
    Add(String),
}

pub(crate) type Worker = thread_worker::Worker<Task, (PathBuf, Vec<FileEvent>)>;

pub(crate) fn start() -> (Worker, WorkerHandle) {
    thread_worker::spawn("vfs", 128, |input_receiver, output_sender| {
        input_receiver
            .map(handle_task)
            .for_each(|it| output_sender.send(it))
    })
}

fn handle_task(task: Task) -> (PathBuf, Vec<FileEvent>) {
    let Task::ScanRoot { path, .. } = task;
    log::debug!("loading {} ...", path.as_path().display());
    let events = load_root(path.as_path());
    log::debug!("... loaded {}", path.as_path().display());
    (path, events)
}

fn load_root(path: &Path) -> Vec<FileEvent> {
    let mut res = Vec::new();
    for entry in WalkDir::new(path) {
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
        if path.extension().and_then(|os| os.to_str()) != Some("rs") {
            continue;
        }
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => {
                log::warn!("watcher error: {}", e);
                continue;
            }
        };
        res.push(FileEvent {
            path: path.to_owned(),
            kind: FileEventKind::Add(text),
        })
    }
    res
}
