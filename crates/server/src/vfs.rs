use std::{
    path::{PathBuf, Path},
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
}

pub fn roots_loader() -> (Sender<PathBuf>, Receiver<(PathBuf, Vec<FileEvent>)>, ThreadWatcher) {
    let (path_sender, path_receiver) = bounded::<PathBuf>(2048);
    let (event_sender, event_receiver) = bounded::<(PathBuf, Vec<FileEvent>)>(1);
    let thread = ThreadWatcher::spawn("roots loader", move || {
        path_receiver
            .into_iter()
            .map(|path| {
                debug!("loading {} ...", path.as_path().display());
                let events = load_root(path.as_path());
                debug!("... loaded {}", path.as_path().display());
                (path, events)
            })
            .for_each(|it| event_sender.send(it))
    });

    (path_sender, event_receiver, thread)
}

fn load_root(path: &Path) -> Vec<FileEvent> {
    let mut res = Vec::new();
    for entry in WalkDir::new(path) {
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
        res.push(FileEvent {
            path: path.to_owned(),
            kind: FileEventKind::Add(text),
        })
    }
    res
}
