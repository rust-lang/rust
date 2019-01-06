use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

use crossbeam_channel::Receiver;
use drop_bomb::DropBomb;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};

pub struct Watcher {
    receiver: Receiver<WatcherChange>,
    watcher: RecommendedWatcher,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

#[derive(Debug)]
pub enum WatcherChange {
    Create(PathBuf),
    Write(PathBuf),
    Remove(PathBuf),
    Rename(PathBuf, PathBuf),
}

impl WatcherChange {
    fn from_debounced_event(ev: DebouncedEvent) -> Option<WatcherChange> {
        match ev {
            DebouncedEvent::NoticeWrite(_)
            | DebouncedEvent::NoticeRemove(_)
            | DebouncedEvent::Chmod(_)
            | DebouncedEvent::Rescan => {
                // ignore
                None
            }
            DebouncedEvent::Create(path) => Some(WatcherChange::Create(path)),
            DebouncedEvent::Write(path) => Some(WatcherChange::Write(path)),
            DebouncedEvent::Remove(path) => Some(WatcherChange::Remove(path)),
            DebouncedEvent::Rename(src, dst) => Some(WatcherChange::Rename(src, dst)),
            DebouncedEvent::Error(err, path) => {
                log::warn!("watch error {}, {:?}", err, path);
                None
            }
        }
    }
}

impl Watcher {
    pub fn start() -> Result<Watcher, Box<std::error::Error>> {
        let (input_sender, input_receiver) = mpsc::channel();
        let watcher = notify::watcher(input_sender, Duration::from_millis(250))?;
        let (output_sender, output_receiver) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || {
            input_receiver
                .into_iter()
                // forward relevant events only
                .filter_map(WatcherChange::from_debounced_event)
                .try_for_each(|change| output_sender.send(change))
                .unwrap()
        });
        Ok(Watcher {
            receiver: output_receiver,
            watcher,
            thread,
            bomb: DropBomb::new(format!("Watcher was not shutdown")),
        })
    }

    pub fn watch(&mut self, root: impl AsRef<Path>) -> Result<(), Box<std::error::Error>> {
        self.watcher.watch(root, RecursiveMode::Recursive)?;
        Ok(())
    }

    pub fn change_receiver(&self) -> &Receiver<WatcherChange> {
        &self.receiver
    }

    pub fn shutdown(mut self) -> thread::Result<()> {
        self.bomb.defuse();
        drop(self.watcher);
        // TODO this doesn't terminate for some reason
        // let res = self.thread.join();
        // match &res {
        //     Ok(()) => log::info!("... Watcher terminated with ok"),
        //     Err(_) => log::error!("... Watcher terminated with err"),
        // }
        // res
        Ok(())
    }
}
