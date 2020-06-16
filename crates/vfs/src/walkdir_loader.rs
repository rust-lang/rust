//! A walkdir-based implementation of `loader::Handle`, which doesn't try to
//! watch files.
use std::convert::TryFrom;

use globset::{Glob, GlobSetBuilder};
use paths::{AbsPath, AbsPathBuf};
use walkdir::WalkDir;

use crate::loader;

#[derive(Debug)]
pub struct WalkdirLoaderHandle {
    // Relative order of fields below is significant.
    sender: crossbeam_channel::Sender<Message>,
    _thread: jod_thread::JoinHandle,
}

enum Message {
    Config(loader::Config),
    Invalidate(AbsPathBuf),
}

impl loader::Handle for WalkdirLoaderHandle {
    fn spawn(sender: loader::Sender) -> WalkdirLoaderHandle {
        let actor = WalkdirLoaderActor { sender };
        let (sender, receiver) = crossbeam_channel::unbounded::<Message>();
        let thread = jod_thread::spawn(move || actor.run(receiver));
        WalkdirLoaderHandle { sender, _thread: thread }
    }
    fn set_config(&mut self, config: loader::Config) {
        self.sender.send(Message::Config(config)).unwrap()
    }
    fn invalidate(&mut self, path: AbsPathBuf) {
        self.sender.send(Message::Invalidate(path)).unwrap();
    }
    fn load_sync(&mut self, path: &AbsPathBuf) -> Option<Vec<u8>> {
        read(path)
    }
}

struct WalkdirLoaderActor {
    sender: loader::Sender,
}

impl WalkdirLoaderActor {
    fn run(mut self, receiver: crossbeam_channel::Receiver<Message>) {
        for msg in receiver {
            match msg {
                Message::Config(config) => {
                    self.send(loader::Message::DidSwitchConfig { n_entries: config.load.len() });
                    for entry in config.load.into_iter() {
                        let files = self.load_entry(entry);
                        self.send(loader::Message::Loaded { files });
                    }
                    drop(config.watch);
                    self.send(loader::Message::DidLoadAllEntries);
                }
                Message::Invalidate(path) => {
                    let contents = read(path.as_path());
                    let files = vec![(path, contents)];
                    self.send(loader::Message::Loaded { files });
                }
            }
        }
    }
    fn load_entry(&mut self, entry: loader::Entry) -> Vec<(AbsPathBuf, Option<Vec<u8>>)> {
        match entry {
            loader::Entry::Files(files) => files
                .into_iter()
                .map(|file| {
                    let contents = read(file.as_path());
                    (file, contents)
                })
                .collect::<Vec<_>>(),
            loader::Entry::Directory { path, globs } => {
                let globset = {
                    let mut builder = GlobSetBuilder::new();
                    for glob in &globs {
                        builder.add(Glob::new(glob).unwrap());
                    }
                    builder.build().unwrap()
                };

                let files = WalkDir::new(path)
                    .into_iter()
                    .filter_map(|it| it.ok())
                    .filter(|it| it.file_type().is_file())
                    .map(|it| it.into_path())
                    .map(|it| AbsPathBuf::try_from(it).unwrap())
                    .filter(|it| globset.is_match(&it));

                files
                    .map(|file| {
                        let contents = read(file.as_path());
                        (file, contents)
                    })
                    .collect()
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
