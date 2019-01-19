use crate::io;
use crossbeam_channel::Sender;
use drop_bomb::DropBomb;
use ignore;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use parking_lot::Mutex;
use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

pub struct Watcher {
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

#[derive(Debug)]
pub enum WatcherChange {
    Create(PathBuf),
    Write(PathBuf),
    Remove(PathBuf),
    Rescan,
}

fn handle_change_event(
    ev: DebouncedEvent,
    sender: &Sender<io::Task>,
    watcher: &Arc<Mutex<Option<RecommendedWatcher>>>,
) -> Result<(), Box<std::error::Error>> {
    match ev {
        DebouncedEvent::NoticeWrite(_)
        | DebouncedEvent::NoticeRemove(_)
        | DebouncedEvent::Chmod(_) => {
            // ignore
        }
        DebouncedEvent::Rescan => {
            sender.send(io::Task::HandleChange(WatcherChange::Rescan))?;
        }
        DebouncedEvent::Create(path) => {
            if path.is_dir() {
                watch_recursive(watcher, &path);
            }
            sender.send(io::Task::HandleChange(WatcherChange::Create(path)))?;
        }
        DebouncedEvent::Write(path) => {
            sender.send(io::Task::HandleChange(WatcherChange::Write(path)))?;
        }
        DebouncedEvent::Remove(path) => {
            sender.send(io::Task::HandleChange(WatcherChange::Remove(path)))?;
        }
        DebouncedEvent::Rename(src, dst) => {
            sender.send(io::Task::HandleChange(WatcherChange::Remove(src)))?;
            sender.send(io::Task::HandleChange(WatcherChange::Create(dst)))?;
        }
        DebouncedEvent::Error(err, path) => {
            // TODO should we reload the file contents?
            log::warn!("watcher error \"{}\", {:?}", err, path);
        }
    }
    Ok(())
}

fn watch_one(watcher: &mut RecommendedWatcher, path: &Path) {
    match watcher.watch(path, RecursiveMode::NonRecursive) {
        Ok(()) => log::debug!("watching \"{}\"", path.display()),
        Err(e) => log::warn!("could not watch \"{}\": {}", path.display(), e),
    }
}

fn watch_recursive(watcher: &Arc<Mutex<Option<RecommendedWatcher>>>, path: &Path) {
    log::debug!("watch_recursive \"{}\"", path.display());
    let mut watcher = watcher.lock();
    let mut watcher = match *watcher {
        Some(ref mut watcher) => watcher,
        None => {
            // watcher has been dropped
            return;
        }
    };
    // TODO it seems path itself isn't checked against ignores
    // check if path should be ignored before walking it
    for res in ignore::Walk::new(path) {
        match res {
            Ok(entry) => {
                if entry.path().is_dir() {
                    watch_one(&mut watcher, entry.path());
                }
            }
            Err(e) => log::warn!("watcher error: {}", e),
        }
    }
}

impl Watcher {
    pub(crate) fn start(
        output_sender: Sender<io::Task>,
    ) -> Result<Watcher, Box<std::error::Error>> {
        let (input_sender, input_receiver) = mpsc::channel();
        let watcher = Arc::new(Mutex::new(Some(notify::watcher(
            input_sender,
            Duration::from_millis(250),
        )?)));
        let w = watcher.clone();
        let thread = thread::spawn(move || {
            input_receiver
                .into_iter()
                // forward relevant events only
                .try_for_each(|change| handle_change_event(change, &output_sender, &w))
                .unwrap()
        });
        Ok(Watcher {
            watcher,
            thread,
            bomb: DropBomb::new(format!("Watcher was not shutdown")),
        })
    }

    pub fn watch(&mut self, root: impl AsRef<Path>) {
        watch_recursive(&self.watcher, root.as_ref());
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
