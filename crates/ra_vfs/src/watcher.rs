use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use drop_bomb::DropBomb;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use crate::{has_rs_extension, io};

pub struct Watcher {
    watcher: RecommendedWatcher,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

#[derive(Debug)]
pub enum WatcherChange {
    Create(PathBuf),
    Write(PathBuf),
    Remove(PathBuf),
    // can this be replaced and use Remove and Create instead?
    Rename(PathBuf, PathBuf),
}

impl WatcherChange {
    fn try_from_debounced_event(ev: DebouncedEvent) -> Option<WatcherChange> {
        match ev {
            DebouncedEvent::NoticeWrite(_)
            | DebouncedEvent::NoticeRemove(_)
            | DebouncedEvent::Chmod(_) => {
                // ignore
                None
            }
            DebouncedEvent::Rescan => {
                // TODO should we rescan the root?
                None
            }
            DebouncedEvent::Create(path) => {
                if has_rs_extension(&path) {
                    Some(WatcherChange::Create(path))
                } else {
                    None
                }
            }
            DebouncedEvent::Write(path) => {
                if has_rs_extension(&path) {
                    Some(WatcherChange::Write(path))
                } else {
                    None
                }
            }
            DebouncedEvent::Remove(path) => {
                if has_rs_extension(&path) {
                    Some(WatcherChange::Remove(path))
                } else {
                    None
                }
            }
            DebouncedEvent::Rename(src, dst) => {
                match (has_rs_extension(&src), has_rs_extension(&dst)) {
                    (true, true) => Some(WatcherChange::Rename(src, dst)),
                    (true, false) => Some(WatcherChange::Remove(src)),
                    (false, true) => Some(WatcherChange::Create(dst)),
                    (false, false) => None,
                }
            }
            DebouncedEvent::Error(err, path) => {
                // TODO should we reload the file contents?
                log::warn!("watch error {}, {:?}", err, path);
                None
            }
        }
    }
}

impl Watcher {
    pub(crate) fn start(
        output_sender: Sender<io::Task>,
    ) -> Result<Watcher, Box<std::error::Error>> {
        let (input_sender, input_receiver) = mpsc::channel();
        let watcher = notify::watcher(input_sender, Duration::from_millis(250))?;
        let thread = thread::spawn(move || {
            input_receiver
                .into_iter()
                // forward relevant events only
                .filter_map(WatcherChange::try_from_debounced_event)
                .try_for_each(|change| output_sender.send(io::Task::WatcherChange(change)))
                .unwrap()
        });
        Ok(Watcher {
            watcher,
            thread,
            bomb: DropBomb::new(format!("Watcher was not shutdown")),
        })
    }

    pub fn watch(&mut self, root: impl AsRef<Path>) -> Result<(), Box<std::error::Error>> {
        self.watcher.watch(root, RecursiveMode::Recursive)?;
        Ok(())
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
