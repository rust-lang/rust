use crate::{io, RootFilter, Roots, VfsRoot};
use crossbeam_channel::Sender;
use drop_bomb::DropBomb;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use parking_lot::Mutex;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};
use walkdir::WalkDir;

#[derive(Debug)]
enum ChangeKind {
    Create,
    Write,
    Remove,
}

const WATCHER_DELAY: Duration = Duration::from_millis(250);

pub(crate) struct Watcher {
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
}

impl Watcher {
    pub(crate) fn start(
        roots: Arc<Roots>,
        output_sender: Sender<io::TaskResult>,
    ) -> Result<Watcher, Box<std::error::Error>> {
        let (input_sender, input_receiver) = mpsc::channel();
        let watcher = Arc::new(Mutex::new(Some(notify::watcher(
            input_sender,
            WATCHER_DELAY,
        )?)));
        let sender = output_sender.clone();
        let watcher_clone = watcher.clone();
        let thread = thread::spawn(move || {
            let worker = WatcherWorker {
                roots,
                watcher: watcher_clone,
                sender,
            };
            input_receiver
                .into_iter()
                // forward relevant events only
                .try_for_each(|change| worker.handle_debounced_event(change))
                .unwrap()
        });
        Ok(Watcher {
            thread,
            watcher,
            bomb: DropBomb::new(format!("Watcher was not shutdown")),
        })
    }

    pub fn watch_root(&mut self, filter: &RootFilter) {
        for res in WalkDir::new(&filter.root)
            .into_iter()
            .filter_entry(filter.entry_filter())
        {
            match res {
                Ok(entry) => {
                    if entry.file_type().is_dir() {
                        watch_one(self.watcher.as_ref(), entry.path());
                    }
                }
                Err(e) => log::warn!("watcher error: {}", e),
            }
        }
    }

    pub fn shutdown(mut self) -> thread::Result<()> {
        self.bomb.defuse();
        drop(self.watcher.lock().take());
        let res = self.thread.join();
        match &res {
            Ok(()) => log::info!("... Watcher terminated with ok"),
            Err(_) => log::error!("... Watcher terminated with err"),
        }
        res
    }
}

struct WatcherWorker {
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
    roots: Arc<Roots>,
    sender: Sender<io::TaskResult>,
}

impl WatcherWorker {
    fn handle_debounced_event(&self, ev: DebouncedEvent) -> Result<(), Box<std::error::Error>> {
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
                self.handle_change(path, ChangeKind::Create);
            }
            DebouncedEvent::Write(path) => {
                self.handle_change(path, ChangeKind::Write);
            }
            DebouncedEvent::Remove(path) => {
                self.handle_change(path, ChangeKind::Remove);
            }
            DebouncedEvent::Rename(src, dst) => {
                self.handle_change(src, ChangeKind::Remove);
                self.handle_change(dst, ChangeKind::Create);
            }
            DebouncedEvent::Error(err, path) => {
                // TODO should we reload the file contents?
                log::warn!("watcher error \"{}\", {:?}", err, path);
            }
        }
        Ok(())
    }

    fn handle_change(&self, path: PathBuf, kind: ChangeKind) {
        if let Err(e) = self.try_handle_change(path, kind) {
            log::warn!("watcher error: {}", e)
        }
    }

    fn try_handle_change(
        &self,
        path: PathBuf,
        kind: ChangeKind,
    ) -> Result<(), Box<std::error::Error>> {
        let (root, rel_path) = match self.roots.find(&path) {
            Some(x) => x,
            None => return Ok(()),
        };
        match kind {
            ChangeKind::Create => {
                if path.is_dir() {
                    self.watch_recursive(&path, root);
                } else {
                    let text = fs::read_to_string(&path)?;
                    self.sender.send(io::TaskResult::AddSingleFile {
                        root,
                        path: rel_path,
                        text,
                    })?
                }
            }
            ChangeKind::Write => {
                let text = fs::read_to_string(&path)?;
                self.sender.send(io::TaskResult::ChangeSingleFile {
                    root,
                    path: rel_path,
                    text,
                })?
            }
            ChangeKind::Remove => self.sender.send(io::TaskResult::RemoveSingleFile {
                root,
                path: rel_path,
            })?,
        }
        Ok(())
    }

    fn watch_recursive(&self, dir: &Path, root: VfsRoot) {
        let filter = &self.roots[root];
        for res in WalkDir::new(dir)
            .into_iter()
            .filter_entry(filter.entry_filter())
        {
            match res {
                Ok(entry) => {
                    if entry.file_type().is_dir() {
                        watch_one(self.watcher.as_ref(), entry.path());
                    } else {
                        // emit only for files otherwise we will cause watch_recursive to be called again with a dir that we are already watching
                        // emit as create because we haven't seen it yet
                        self.handle_change(entry.path().to_path_buf(), ChangeKind::Create);
                    }
                }
                Err(e) => log::warn!("watcher error: {}", e),
            }
        }
    }
}

fn watch_one(watcher: &Mutex<Option<RecommendedWatcher>>, dir: &Path) {
    if let Some(watcher) = watcher.lock().as_mut() {
        match watcher.watch(dir, RecursiveMode::NonRecursive) {
            Ok(()) => log::debug!("watching \"{}\"", dir.display()),
            Err(e) => log::warn!("could not watch \"{}\": {}", dir.display(), e),
        }
    }
}
