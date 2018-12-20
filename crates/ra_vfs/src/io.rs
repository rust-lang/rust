use std::{
    fmt,
    fs,
    path::{Path, PathBuf},
};

use walkdir::{DirEntry, WalkDir};
use thread_worker::{WorkerHandle};
use relative_path::RelativePathBuf;

use crate::VfsRoot;

pub(crate) struct Task {
    pub(crate) root: VfsRoot,
    pub(crate) path: PathBuf,
    pub(crate) filter: Box<Fn(&DirEntry) -> bool + Send>,
}

pub struct TaskResult {
    pub(crate) root: VfsRoot,
    pub(crate) files: Vec<(RelativePathBuf, String)>,
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
            .map(handle_task)
            .for_each(|it| output_sender.send(it))
    })
}

fn handle_task(task: Task) -> TaskResult {
    let Task { root, path, filter } = task;
    log::debug!("loading {} ...", path.as_path().display());
    let files = load_root(path.as_path(), &*filter);
    log::debug!("... loaded {}", path.as_path().display());
    TaskResult { root, files }
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
        let path = RelativePathBuf::from_path(path.strip_prefix(root).unwrap()).unwrap();
        res.push((path.to_owned(), text))
    }
    res
}
