use std::{
    path::PathBuf,
    thread,
    fs,
};

use crossbeam_channel::{Sender, Receiver, bounded};
use drop_bomb::DropBomb;
use walkdir::WalkDir;

use Result;


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

pub struct Watcher {
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

impl Watcher {
    pub fn stop(mut self) -> Result<()> {
        self.bomb.defuse();
        self.thread.join()
            .map_err(|_| format_err!("file watcher died"))
    }
}

pub fn watch(roots: Vec<PathBuf>) -> (Receiver<Vec<FileEvent>>, Watcher) {
    let (sender, receiver) = bounded(16);
    let thread = thread::spawn(move || run(roots, sender));
    (receiver, Watcher {
        thread,
        bomb: DropBomb::new("Watcher should be stopped explicitly"),
    })
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
