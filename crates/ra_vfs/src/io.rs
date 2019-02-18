use std::{
    fs,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    time::Duration,
};
use crossbeam_channel::{Sender, unbounded, RecvError, select};
use relative_path::RelativePathBuf;
use walkdir::WalkDir;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as _Watcher};

use crate::{Roots, VfsRoot, VfsTask};

pub(crate) enum Task {
    AddRoot { root: VfsRoot },
}

/// `TaskResult` transfers files read on the IO thread to the VFS on the main
/// thread.
#[derive(Debug)]
pub(crate) enum TaskResult {
    /// Emitted when we've recursively scanned a source root during the initial
    /// load.
    BulkLoadRoot { root: VfsRoot, files: Vec<(RelativePathBuf, String)> },
    /// Emitted when we've noticed that a single file has changed.
    ///
    /// Note that this by design does not distinguish between
    /// create/delete/write events, and instead specifies the *current* state of
    /// the file. The idea is to guarantee that in the quiescent state the sum
    /// of all results equals to the current state of the file system, while
    /// allowing to skip intermediate events in non-quiescent states.
    SingleFile { root: VfsRoot, path: RelativePathBuf, text: Option<String> },
}

/// The kind of raw notification we've received from the notify library.
///
/// Note that these are not necessary 100% precise (for example we might receive
/// `Create` instead of `Write`, see #734), but we try do distinguish `Create`s
/// to implement recursive watching of directories.
#[derive(Debug)]
enum ChangeKind {
    Create,
    Write,
    Remove,
}

const WATCHER_DELAY: Duration = Duration::from_millis(250);

pub(crate) type Worker = thread_worker::Worker<Task, VfsTask>;
pub(crate) fn start(roots: Arc<Roots>) -> Worker {
    // This is a pretty elaborate setup of threads & channels! It is
    // explained by the following concerns:
    //    * we need to burn a thread translating from notify's mpsc to
    //      crossbeam_channel.
    //    * we want to read all files from a single thread, to guarantee that
    //      we always get fresher versions and never go back in time.
    //    * we want to tear down everything neatly during shutdown.
    Worker::spawn(
        "vfs",
        128,
        // This are the channels we use to communicate with outside world.
        // If `input_receiver` is closed we need to tear ourselves down.
        // `output_sender` should not be closed unless the parent died.
        move |input_receiver, output_sender| {
            // Make sure that the destruction order is
            //
            // * notify_sender
            // * _thread
            // * watcher_sender
            //
            // this is required to avoid deadlocks.

            // These are the corresponding crossbeam channels
            let (watcher_sender, watcher_receiver) = unbounded();
            let _thread;
            {
                // These are `std` channels notify will send events to
                let (notify_sender, notify_receiver) = mpsc::channel();

                let mut watcher = notify::watcher(notify_sender, WATCHER_DELAY)
                    .map_err(|e| log::error!("failed to spawn notify {}", e))
                    .ok();
                // Start a silly thread to transform between two channels
                _thread = thread_worker::ScopedThread::spawn("notify-convertor", move || {
                    notify_receiver
                        .into_iter()
                        .for_each(|event| convert_notify_event(event, &watcher_sender))
                });

                // Process requests from the called or notifications from
                // watcher until the caller says stop.
                loop {
                    select! {
                        // Received request from the caller. If this channel is
                        // closed, we should shutdown everything.
                        recv(input_receiver) -> t => match t {
                            Err(RecvError) => {
                                drop(input_receiver);
                                break
                            },
                            Ok(Task::AddRoot { root }) => {
                                watch_root(watcher.as_mut(), &output_sender, &*roots, root);
                            }
                        },
                        // Watcher send us changes. If **this** channel is
                        // closed, the watcher has died, which indicates a bug
                        // -- escalate!
                        recv(watcher_receiver) -> event => match event {
                            Err(RecvError) => panic!("watcher is dead"),
                            Ok((path, change)) => {
                                handle_change(watcher.as_mut(), &output_sender, &*roots, path, change);
                            }
                        },
                    }
                }
            }
            // Drain pending events: we are not interested in them anyways!
            watcher_receiver.into_iter().for_each(|_| ());
        },
    )
}

fn watch_root(
    watcher: Option<&mut RecommendedWatcher>,
    sender: &Sender<VfsTask>,
    roots: &Roots,
    root: VfsRoot,
) {
    let root_path = roots.path(root);
    log::debug!("loading {} ...", root_path.display());
    let files = watch_recursive(watcher, root_path, roots, root)
        .into_iter()
        .filter_map(|path| {
            let abs_path = path.to_path(&root_path);
            let text = read_to_string(&abs_path)?;
            Some((path, text))
        })
        .collect();
    let res = TaskResult::BulkLoadRoot { root, files };
    sender.send(VfsTask(res)).unwrap();
    log::debug!("... loaded {}", root_path.display());
}

fn convert_notify_event(event: DebouncedEvent, sender: &Sender<(PathBuf, ChangeKind)>) {
    // forward relevant events only
    match event {
        DebouncedEvent::NoticeWrite(_)
        | DebouncedEvent::NoticeRemove(_)
        | DebouncedEvent::Chmod(_) => {
            // ignore
        }
        DebouncedEvent::Rescan => {
            // TODO: rescan all roots
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
            // TODO: should we reload the file contents?
            log::warn!("watcher error \"{}\", {:?}", err, path);
        }
    }
}

fn handle_change(
    watcher: Option<&mut RecommendedWatcher>,
    sender: &Sender<VfsTask>,
    roots: &Roots,
    path: PathBuf,
    kind: ChangeKind,
) {
    let (root, rel_path) = match roots.find(&path) {
        None => return,
        Some(it) => it,
    };
    match kind {
        ChangeKind::Create => {
            let mut paths = Vec::new();
            if path.is_dir() {
                paths.extend(watch_recursive(watcher, &path, roots, root));
            } else {
                paths.push(rel_path);
            }
            paths
                .into_iter()
                .try_for_each(|rel_path| {
                    let abs_path = rel_path.to_path(&roots.path(root));
                    let text = read_to_string(&abs_path);
                    let res = TaskResult::SingleFile { root, path: rel_path, text };
                    sender.send(VfsTask(res))
                })
                .unwrap()
        }
        ChangeKind::Write | ChangeKind::Remove => {
            let text = read_to_string(&path);
            let res = TaskResult::SingleFile { root, path: rel_path, text };
            sender.send(VfsTask(res)).unwrap();
        }
    }
}

fn watch_recursive(
    mut watcher: Option<&mut RecommendedWatcher>,
    dir: &Path,
    roots: &Roots,
    root: VfsRoot,
) -> Vec<RelativePathBuf> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir)
        .into_iter()
        .filter_entry(|it| roots.contains(root, it.path()).is_some())
        .filter_map(|it| it.map_err(|e| log::warn!("watcher error: {}", e)).ok())
    {
        if entry.file_type().is_dir() {
            if let Some(watcher) = &mut watcher {
                watch_one(watcher, entry.path());
            }
        } else {
            let path = roots.contains(root, entry.path()).unwrap();
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
    fs::read_to_string(&path).map_err(|e| log::warn!("failed to read file {}", e)).ok()
}
