use std::{
    fs,
    thread,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    time::Duration,
};
use crossbeam_channel::{Receiver, Sender, unbounded, RecvError, select};
use relative_path::RelativePathBuf;
use thread_worker::WorkerHandle;
use walkdir::WalkDir;
use parking_lot::Mutex;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as _Watcher};

use crate::{RootConfig, Roots, VfsRoot};

type Result<T> = std::result::Result<T, crossbeam_channel::SendError<TaskResult>>;

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
        // This is a pretty elaborate setup of threads & channels! It is
        // explained by the following concerns:

        //    * we need to burn a thread translating from notify's mpsc to
        //      crossbeam_channel.
        //    * we want to read all files from a single thread, to gurantee that
        //      we always get fresher versions and never go back in time.
        //    * we want to tear down everything neatly during shutdown.
        let (worker, worker_handle) = thread_worker::spawn(
            "vfs",
            128,
            // This are the channels we use to communicate with outside world.
            // If `input_receiver` is closed we need to tear ourselves down.
            // `output_sender` should not be closed unless the parent died.
            move |input_receiver, output_sender| {
                // These are `std` channels notify will send events to
                let (notify_sender, notify_receiver) = mpsc::channel();
                // These are the corresponding crossbeam channels
                let (watcher_sender, watcher_receiver) = unbounded();
                let watcher = notify::watcher(notify_sender, WATCHER_DELAY)
                    .map_err(|e| log::error!("failed to spawn notify {}", e))
                    .ok();
                let ctx = WatcherCtx {
                    roots,
                    watcher: Arc::new(Mutex::new(watcher)),
                    sender: output_sender,
                };
                let thread = thread::spawn(move || {
                    let _ = notify_receiver
                        .into_iter()
                        // forward relevant events only
                        .for_each(|event| convert_notify_event(event, &watcher_sender));
                });

                loop {
                    select! {
                        // Received request from the caller. If this channel is
                        // closed, we should shutdown everything.
                        recv(input_receiver) -> t => match t {
                            Err(RecvError) => break,
                            Ok(Task::AddRoot { root, config }) => watch_root(&ctx, root, Arc::clone(&config)),
                        },
                        // Watcher send us changes. If **this** channel is
                        // closed, the watcher has died, which indicates a bug
                        // -- escalate!
                        recv(watcher_receiver) -> event => match event {
                            Err(RecvError) => panic!("watcher is dead"),
                            Ok((path, change)) => WatcherCtx::handle_change(&ctx, path, change).unwrap(),
                        },
                    }
                }
                drop(ctx.watcher.lock().take());
                drop(ctx);
                let res2 = thread.join();
                match &res2 {
                    Ok(()) => log::info!("... Watcher terminated with ok"),
                    Err(_) => log::error!("... Watcher terminated with err"),
                }
                res2.unwrap();
            },
        );
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

fn watch_root(woker: &WatcherCtx, root: VfsRoot, config: Arc<RootConfig>) {
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
        .send(TaskResult::BulkLoadRoot { root, files })
        .unwrap();
    log::debug!("... loaded {}", config.root.as_path().display());
}

#[derive(Clone)]
struct WatcherCtx {
    roots: Arc<Roots>,
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
    sender: Sender<TaskResult>,
}

fn convert_notify_event(event: DebouncedEvent, sender: &Sender<(PathBuf, ChangeKind)>) {
    match event {
        DebouncedEvent::NoticeWrite(_)
        | DebouncedEvent::NoticeRemove(_)
        | DebouncedEvent::Chmod(_) => {
            // ignore
        }
        DebouncedEvent::Rescan => {
            // TODO rescan all roots
        }
        DebouncedEvent::Create(path) => {
            sender.send((path, ChangeKind::Create)).unwrap();
        }
        DebouncedEvent::Write(path) => {
            sender.send((path, ChangeKind::Write)).unwrap();
        }
        DebouncedEvent::Remove(path) => {
            sender.send((path, ChangeKind::Remove)).unwrap();
        }
        DebouncedEvent::Rename(src, dst) => {
            sender.send((src, ChangeKind::Remove)).unwrap();
            sender.send((dst, ChangeKind::Create)).unwrap();
        }
        DebouncedEvent::Error(err, path) => {
            // TODO should we reload the file contents?
            log::warn!("watcher error \"{}\", {:?}", err, path);
        }
    }
}

impl WatcherCtx {
    fn handle_change(&self, path: PathBuf, kind: ChangeKind) -> Result<()> {
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
