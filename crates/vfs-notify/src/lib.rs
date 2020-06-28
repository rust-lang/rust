//! An implementation of `loader::Handle`, based on `walkdir` and `notify`.
//!
//! The file watching bits here are untested and quite probably buggy. For this
//! reason, by default we don't watch files and rely on editor's file watching
//! capabilities.
//!
//! Hopefully, one day a reliable file watching/walking crate appears on
//! crates.io, and we can reduce this to trivial glue code.
mod include;

use std::convert::{TryFrom, TryInto};

use crossbeam_channel::{select, unbounded, Receiver, Sender};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashSet;
use vfs::loader;
use walkdir::WalkDir;

use crate::include::Include;

#[derive(Debug)]
pub struct NotifyHandle {
    // Relative order of fields below is significant.
    sender: Sender<Message>,
    thread: jod_thread::JoinHandle,
}

#[derive(Debug)]
enum Message {
    Config(loader::Config),
    Invalidate(AbsPathBuf),
}

impl loader::Handle for NotifyHandle {
    fn spawn(sender: loader::Sender) -> NotifyHandle {
        let actor = NotifyActor::new(sender);
        let (sender, receiver) = unbounded::<Message>();
        let thread = jod_thread::spawn(move || actor.run(receiver));
        NotifyHandle { sender, thread }
    }
    fn set_config(&mut self, config: loader::Config) {
        self.sender.send(Message::Config(config)).unwrap()
    }
    fn invalidate(&mut self, path: AbsPathBuf) {
        self.sender.send(Message::Invalidate(path)).unwrap();
    }
    fn load_sync(&mut self, path: &AbsPath) -> Option<Vec<u8>> {
        read(path)
    }
}

type NotifyEvent = notify::Result<notify::Event>;

struct NotifyActor {
    sender: loader::Sender,
    config: Vec<(AbsPathBuf, Include, bool)>,
    watched_paths: FxHashSet<AbsPathBuf>,
    // Drop order of fields bellow is significant,
    watcher: Option<RecommendedWatcher>,
    watcher_receiver: Receiver<NotifyEvent>,
}

#[derive(Debug)]
enum Event {
    Message(Message),
    NotifyEvent(NotifyEvent),
}

impl NotifyActor {
    fn new(sender: loader::Sender) -> NotifyActor {
        let (watcher_sender, watcher_receiver) = unbounded();
        let watcher = log_notify_error(Watcher::new_immediate(move |event| {
            watcher_sender.send(event).unwrap()
        }));

        NotifyActor {
            sender,
            config: Vec::new(),
            watched_paths: FxHashSet::default(),
            watcher,
            watcher_receiver,
        }
    }
    fn next_event(&self, receiver: &Receiver<Message>) -> Option<Event> {
        select! {
            recv(receiver) -> it => it.ok().map(Event::Message),
            recv(&self.watcher_receiver) -> it => Some(Event::NotifyEvent(it.unwrap())),
        }
    }
    fn run(mut self, inbox: Receiver<Message>) {
        while let Some(event) = self.next_event(&inbox) {
            log::debug!("vfs-notify event: {:?}", event);
            match event {
                Event::Message(msg) => match msg {
                    Message::Config(config) => {
                        let n_total = config.load.len();
                        self.send(loader::Message::Progress { n_total, n_done: 0 });

                        self.unwatch_all();
                        self.config.clear();

                        for (i, entry) in config.load.into_iter().enumerate() {
                            let watch = config.watch.contains(&i);
                            let files = self.load_entry(entry, watch);
                            self.send(loader::Message::Loaded { files });
                            self.send(loader::Message::Progress { n_total, n_done: i + 1 });
                        }
                        self.config.sort_by(|x, y| x.0.cmp(&y.0));
                    }
                    Message::Invalidate(path) => {
                        let contents = read(path.as_path());
                        let files = vec![(path, contents)];
                        self.send(loader::Message::Loaded { files });
                    }
                },
                Event::NotifyEvent(event) => {
                    if let Some(event) = log_notify_error(event) {
                        let files = event
                            .paths
                            .into_iter()
                            .map(|path| AbsPathBuf::try_from(path).unwrap())
                            .filter_map(|path| {
                                let is_dir = path.is_dir();
                                let is_file = path.is_file();

                                let config_idx =
                                    match self.config.binary_search_by(|it| it.0.cmp(&path)) {
                                        Ok(it) => it,
                                        Err(it) => it.saturating_sub(1),
                                    };
                                let include = self.config.get(config_idx).and_then(|it| {
                                    let rel_path = path.strip_prefix(&it.0)?;
                                    Some((rel_path, &it.1))
                                });

                                if let Some((rel_path, include)) = include {
                                    if is_dir && include.exclude_dir(&rel_path)
                                        || is_file && !include.include_file(&rel_path)
                                    {
                                        return None;
                                    }
                                }

                                if is_dir {
                                    self.watch(path);
                                    return None;
                                }
                                if !is_file {
                                    return None;
                                }
                                let contents = read(&path);
                                Some((path, contents))
                            })
                            .collect();
                        self.send(loader::Message::Loaded { files })
                    }
                }
            }
        }
    }
    fn load_entry(
        &mut self,
        entry: loader::Entry,
        watch: bool,
    ) -> Vec<(AbsPathBuf, Option<Vec<u8>>)> {
        match entry {
            loader::Entry::Files(files) => files
                .into_iter()
                .map(|file| {
                    if watch {
                        self.watch(file.clone())
                    }
                    let contents = read(file.as_path());
                    (file, contents)
                })
                .collect::<Vec<_>>(),
            loader::Entry::Directory { path, include } => {
                let include = Include::new(include);
                self.config.push((path.clone(), include.clone(), watch));

                let files = WalkDir::new(&path)
                    .into_iter()
                    .filter_entry(|entry| {
                        let abs_path: &AbsPath = entry.path().try_into().unwrap();
                        match abs_path.strip_prefix(&path) {
                            Some(rel_path) => {
                                !(entry.file_type().is_dir() && include.exclude_dir(rel_path))
                            }
                            None => false,
                        }
                    })
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| {
                        let is_dir = entry.file_type().is_dir();
                        let is_file = entry.file_type().is_file();
                        let abs_path = AbsPathBuf::try_from(entry.into_path()).unwrap();
                        if is_dir && watch {
                            self.watch(abs_path.clone());
                        }
                        let rel_path = abs_path.strip_prefix(&path)?;
                        if is_file && include.include_file(&rel_path) {
                            Some(abs_path)
                        } else {
                            None
                        }
                    });

                files
                    .map(|file| {
                        let contents = read(file.as_path());
                        (file, contents)
                    })
                    .collect()
            }
        }
    }

    fn watch(&mut self, path: AbsPathBuf) {
        if let Some(watcher) = &mut self.watcher {
            log_notify_error(watcher.watch(&path, RecursiveMode::NonRecursive));
            self.watched_paths.insert(path);
        }
    }
    fn unwatch_all(&mut self) {
        if let Some(watcher) = &mut self.watcher {
            for path in self.watched_paths.drain() {
                log_notify_error(watcher.unwatch(path));
            }
        }
    }
    fn send(&mut self, msg: loader::Message) {
        (self.sender)(msg)
    }
}

fn read(path: &AbsPath) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn log_notify_error<T>(res: notify::Result<T>) -> Option<T> {
    res.map_err(|err| log::warn!("notify error: {}", err)).ok()
}
