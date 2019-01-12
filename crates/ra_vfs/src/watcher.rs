use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

use crate::io;
use crossbeam_channel::Sender;
use drop_bomb::DropBomb;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};

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
    Rescan,
}

fn send_change_events(
    ev: DebouncedEvent,
    sender: &Sender<io::Task>,
) -> Result<(), Box<std::error::Error>> {
    match ev {
        DebouncedEvent::NoticeWrite(_)
        | DebouncedEvent::NoticeRemove(_)
        | DebouncedEvent::Chmod(_) => {
            // ignore
        }
        DebouncedEvent::Rescan => {
            sender.send(io::Task::LoadChange(WatcherChange::Rescan))?;
        }
        DebouncedEvent::Create(path) => {
            sender.send(io::Task::LoadChange(WatcherChange::Create(path)))?;
        }
        DebouncedEvent::Write(path) => {
            sender.send(io::Task::LoadChange(WatcherChange::Write(path)))?;
        }
        DebouncedEvent::Remove(path) => {
            sender.send(io::Task::LoadChange(WatcherChange::Remove(path)))?;
        }
        DebouncedEvent::Rename(src, dst) => {
            sender.send(io::Task::LoadChange(WatcherChange::Remove(src)))?;
            sender.send(io::Task::LoadChange(WatcherChange::Create(dst)))?;
        }
        DebouncedEvent::Error(err, path) => {
            // TODO should we reload the file contents?
            log::warn!("watcher error {}, {:?}", err, path);
        }
    }
    Ok(())
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
                .try_for_each(|change| send_change_events(change, &output_sender))
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
        // TODO this doesn't terminate because of a buf in `notify`
        // uncomment when https://github.com/passcod/notify/pull/170 is released
        // let res = self.thread.join();
        // match &res {
        //     Ok(()) => log::info!("... Watcher terminated with ok"),
        //     Err(_) => log::error!("... Watcher terminated with err"),
        // }
        // res
        Ok(())
    }
}
