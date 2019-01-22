use crate::{io, RootFilter};
use crossbeam_channel::Sender;
use drop_bomb::DropBomb;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};
use walkdir::WalkDir;

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

const WATCHER_DELAY: Duration = Duration::from_millis(250);

pub(crate) struct Watcher {
    watcher: RecommendedWatcher,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
    sender: Sender<io::Task>,
}

impl Watcher {
    pub(crate) fn start(
        output_sender: Sender<io::Task>,
    ) -> Result<Watcher, Box<std::error::Error>> {
        let (input_sender, input_receiver) = mpsc::channel();
        let watcher = notify::watcher(input_sender, WATCHER_DELAY)?;
        let sender = output_sender.clone();
        let thread = thread::spawn(move || {
            input_receiver
                .into_iter()
                // forward relevant events only
                .try_for_each(|change| handle_change_event(change, &output_sender))
                .unwrap()
        });
        Ok(Watcher {
            watcher,
            thread,
            sender,
            bomb: DropBomb::new(format!("Watcher was not shutdown")),
        })
    }

    pub fn watch_recursive(&mut self, dir: &Path, filter: &RootFilter, emit_for_contents: bool) {
        for res in WalkDir::new(dir)
            .into_iter()
            .filter_entry(|entry| filter.can_contain(entry.path()).is_some())
        {
            match res {
                Ok(entry) => {
                    if entry.path().is_dir() {
                        match self
                            .watcher
                            .watch(entry.path(), RecursiveMode::NonRecursive)
                        {
                            Ok(()) => log::debug!("watching \"{}\"", entry.path().display()),
                            Err(e) => {
                                log::warn!("could not watch \"{}\": {}", entry.path().display(), e)
                            }
                        }
                    } else {
                        if emit_for_contents && entry.depth() > 0 {
                            // emit only for files otherwise we will cause watch_recursive to be called again with a dir that we are already watching
                            // emit as create because we haven't seen it yet
                            if let Err(e) =
                                self.sender
                                    .send(io::Task::HandleChange(WatcherChange::Create(
                                        entry.path().to_path_buf(),
                                    )))
                            {
                                log::warn!("watcher error: {}", e)
                            }
                        }
                    }
                }
                Err(e) => log::warn!("watcher error: {}", e),
            }
        }
    }

    pub fn shutdown(mut self) -> thread::Result<()> {
        self.bomb.defuse();
        drop(self.watcher);
        let res = self.thread.join();
        match &res {
            Ok(()) => log::info!("... Watcher terminated with ok"),
            Err(_) => log::error!("... Watcher terminated with err"),
        }
        res
    }
}
