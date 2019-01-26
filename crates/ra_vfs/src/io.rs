use std::{
    fs,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};
use crossbeam_channel::{Receiver, Sender, SendError};
use relative_path::RelativePathBuf;
use thread_worker::WorkerHandle;
use walkdir::WalkDir;
use parking_lot::Mutex;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as _Watcher};

use crate::{RootConfig, Roots, VfsRoot};

pub(crate) enum Task {
    AddRoot {
        root: VfsRoot,
        config: Arc<RootConfig>,
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

#[derive(Debug)]
enum ChangeKind {
    Create,
    Write,
    Remove,
}

const WATCHER_DELAY: Duration = Duration::from_millis(250);

pub(crate) struct Worker {
    worker: thread_worker::Worker<Task, TaskResult>,
    worker_handle: WorkerHandle,
}

impl Worker {
    pub(crate) fn start(roots: Arc<Roots>) -> Worker {
        let (worker, worker_handle) =
            thread_worker::spawn("vfs", 128, move |input_receiver, output_sender| {
                let (notify_sender, notify_receiver) = mpsc::channel();
                let watcher = notify::watcher(notify_sender, WATCHER_DELAY)
                    .map_err(|e| log::error!("failed to spawn notify {}", e))
                    .ok();
                let ctx = WatcherCtx {
                    roots,
                    watcher: Arc::new(Mutex::new(watcher)),
                    sender: output_sender,
                };
                let thread = thread::spawn({
                    let ctx = ctx.clone();
                    move || {
                        let _ = notify_receiver
                            .into_iter()
                            // forward relevant events only
                            .try_for_each(|change| ctx.handle_debounced_event(change));
                    }
                });
                let res1 = input_receiver.into_iter().try_for_each(|t| match t {
                    Task::AddRoot { root, config } => watch_root(&ctx, root, Arc::clone(&config)),
                });
                drop(ctx.watcher.lock().take());
                drop(ctx);
                let res2 = thread.join();
                match &res2 {
                    Ok(()) => log::info!("... Watcher terminated with ok"),
                    Err(_) => log::error!("... Watcher terminated with err"),
                }
                res1.unwrap();
                res2.unwrap();
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

fn watch_root(
    woker: &WatcherCtx,
    root: VfsRoot,
    config: Arc<RootConfig>,
) -> Result<(), SendError<TaskResult>> {
    let mut guard = woker.watcher.lock();
    log::debug!("loading {} ...", config.root.as_path().display());
    let files = watch_recursive(guard.as_mut(), config.root.as_path(), &*config)
        .into_iter()
        .filter_map(|path| {
            let abs_path = path.to_path(&config.root);
            let text = read_to_string(&abs_path)?;
            Some((path, text))
        })
        .collect();
    woker
        .sender
        .send(TaskResult::BulkLoadRoot { root, files })?;
    log::debug!("... loaded {}", config.root.as_path().display());
    Ok(())
}

#[derive(Clone)]
struct WatcherCtx {
    roots: Arc<Roots>,
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
    sender: Sender<TaskResult>,
}

impl WatcherCtx {
    fn handle_debounced_event(&self, ev: DebouncedEvent) -> Result<(), SendError<TaskResult>> {
        match ev {
            DebouncedEvent::NoticeWrite(_)
            | DebouncedEvent::NoticeRemove(_)
            | DebouncedEvent::Chmod(_) => {
                // ignore
            }
            DebouncedEvent::Rescan => {
                // TODO rescan all roots
            }
            DebouncedEvent::Create(path) => {
                self.handle_change(path, ChangeKind::Create)?;
            }
            DebouncedEvent::Write(path) => {
                self.handle_change(path, ChangeKind::Write)?;
            }
            DebouncedEvent::Remove(path) => {
                self.handle_change(path, ChangeKind::Remove)?;
            }
            DebouncedEvent::Rename(src, dst) => {
                self.handle_change(src, ChangeKind::Remove)?;
                self.handle_change(dst, ChangeKind::Create)?;
            }
            DebouncedEvent::Error(err, path) => {
                // TODO should we reload the file contents?
                log::warn!("watcher error \"{}\", {:?}", err, path);
            }
        }
        Ok(())
    }

    fn handle_change(&self, path: PathBuf, kind: ChangeKind) -> Result<(), SendError<TaskResult>> {
        let (root, rel_path) = match self.roots.find(&path) {
            None => return Ok(()),
            Some(it) => it,
        };
        let config = &self.roots[root];
        match kind {
            ChangeKind::Create => {
                let mut paths = Vec::new();
                if path.is_dir() {
                    let mut guard = self.watcher.lock();
                    paths.extend(watch_recursive(guard.as_mut(), &path, &config));
                } else {
                    paths.push(rel_path);
                }
                paths
                    .into_iter()
                    .filter_map(|rel_path| {
                        let abs_path = rel_path.to_path(&config.root);
                        let text = read_to_string(&abs_path)?;
                        Some((rel_path, text))
                    })
                    .try_for_each(|(path, text)| {
                        self.sender
                            .send(TaskResult::AddSingleFile { root, path, text })
                    })?
            }
            ChangeKind::Write => {
                if let Some(text) = read_to_string(&path) {
                    self.sender.send(TaskResult::ChangeSingleFile {
                        root,
                        path: rel_path,
                        text,
                    })?;
                }
            }
            ChangeKind::Remove => self.sender.send(TaskResult::RemoveSingleFile {
                root,
                path: rel_path,
            })?,
        }
        Ok(())
    }
}

fn watch_recursive(
    mut watcher: Option<&mut RecommendedWatcher>,
    dir: &Path,
    config: &RootConfig,
) -> Vec<RelativePathBuf> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir)
        .into_iter()
        .filter_entry(|it| config.contains(it.path()).is_some())
        .filter_map(|it| it.map_err(|e| log::warn!("watcher error: {}", e)).ok())
    {
        if entry.file_type().is_dir() {
            if let Some(watcher) = &mut watcher {
                watch_one(watcher, entry.path());
            }
        } else {
            let path = config.contains(entry.path()).unwrap();
            files.push(path.to_owned());
        }
    }
    files
}

fn watch_one(watcher: &mut RecommendedWatcher, dir: &Path) {
    match watcher.watch(dir, RecursiveMode::NonRecursive) {
        Ok(()) => log::debug!("watching \"{}\"", dir.display()),
        Err(e) => log::warn!("could not watch \"{}\": {}", dir.display(), e),
    }
}

fn read_to_string(path: &Path) -> Option<String> {
    fs::read_to_string(&path)
        .map_err(|e| log::warn!("failed to read file {}", e))
        .ok()
}
