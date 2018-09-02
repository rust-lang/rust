use std::{
    path::PathBuf,
    fs,
};

use crossbeam_channel::{Sender, Receiver, bounded};
use walkdir::WalkDir;

use {
    thread_watcher::ThreadWatcher,
};


#[derive(Debug)]
pub struct FileEvent {
    pub path: PathBuf,
    pub kind: FileEventKind,
}

#[derive(Debug)]
pub enum FileEventKind {
    Add(String),
    #[allow(unused)]
    Remove,
}

pub fn watch(roots: Vec<PathBuf>) -> (Receiver<Vec<FileEvent>>, ThreadWatcher) {
    let (sender, receiver) = bounded(16);
    let watcher = ThreadWatcher::spawn("vfs", move || run(roots, sender));
    (receiver, watcher)
}

fn run(roots: Vec<PathBuf>, sender: Sender<Vec<FileEvent>>) {
    for root in roots {
        let mut events = Vec::new();
        for entry in WalkDir::new(root.as_path()) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    warn!("watcher error: {}", e);
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|os| os.to_str()) != Some("rs") {
                continue;
            }
            let text = match fs::read_to_string(path) {
                Ok(text) => text,
                Err(e) => {
                    warn!("watcher error: {}", e);
                    continue;
                }
            };
            events.push(FileEvent {
                path: path.to_owned(),
                kind: FileEventKind::Add(text),
            })
        }
        sender.send(events)
    }
}
