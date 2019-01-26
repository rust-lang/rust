use std::{fs, sync::Arc, thread};

use crossbeam_channel::{Receiver, Sender};
use relative_path::RelativePathBuf;
use thread_worker::WorkerHandle;
use walkdir::WalkDir;

mod watcher;
use watcher::Watcher;

use crate::{RootFilter, Roots, VfsRoot};

pub(crate) enum Task {
    AddRoot {
        root: VfsRoot,
        filter: Arc<RootFilter>,
    },
}

#[derive(Debug)]
pub enum TaskResult {
    BulkLoadRoot {
        root: VfsRoot,
        files: Vec<(RelativePathBuf, String)>,
    },
    AddSingleFile {
        root: VfsRoot,
        path: RelativePathBuf,
        text: String,
    },
    ChangeSingleFile {
        root: VfsRoot,
        path: RelativePathBuf,
        text: String,
    },
    RemoveSingleFile {
        root: VfsRoot,
        path: RelativePathBuf,
    },
}

pub(crate) struct Worker {
    worker: thread_worker::Worker<Task, TaskResult>,
    worker_handle: WorkerHandle,
}

impl Worker {
    pub(crate) fn start(roots: Arc<Roots>) -> Worker {
        let (worker, worker_handle) =
            thread_worker::spawn("vfs", 128, move |input_receiver, output_sender| {
                let mut watcher = match Watcher::start(roots, output_sender.clone()) {
                    Ok(w) => Some(w),
                    Err(e) => {
                        log::error!("could not start watcher: {}", e);
                        None
                    }
                };
                let res = input_receiver
                    .into_iter()
                    .filter_map(|t| handle_task(t, &mut watcher))
                    .try_for_each(|it| output_sender.send(it));
                if let Some(watcher) = watcher {
                    let _ = watcher.shutdown();
                }
                res.unwrap()
            });
        Worker {
            worker,
            worker_handle,
        }
    }

    pub(crate) fn sender(&self) -> &Sender<Task> {
        &self.worker.inp
    }

    pub(crate) fn receiver(&self) -> &Receiver<TaskResult> {
        &self.worker.out
    }

    pub(crate) fn shutdown(self) -> thread::Result<()> {
        let _ = self.worker.shutdown();
        self.worker_handle.shutdown()
    }
}

fn handle_task(task: Task, watcher: &mut Option<Watcher>) -> Option<TaskResult> {
    match task {
        Task::AddRoot { root, filter } => {
            if let Some(watcher) = watcher {
                watcher.watch_root(&filter)
            }
            log::debug!("loading {} ...", filter.root.as_path().display());
            let files = load_root(filter.as_ref());
            log::debug!("... loaded {}", filter.root.as_path().display());
            Some(TaskResult::BulkLoadRoot { root, files })
        }
    }
}

fn load_root(filter: &RootFilter) -> Vec<(RelativePathBuf, String)> {
    let mut res = Vec::new();
    for entry in WalkDir::new(&filter.root)
        .into_iter()
        .filter_entry(filter.entry_filter())
    {
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
        let path = RelativePathBuf::from_path(path.strip_prefix(&filter.root).unwrap()).unwrap();
        res.push((path.to_owned(), text))
    }
    res
}
