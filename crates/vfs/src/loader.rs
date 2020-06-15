//! Object safe interface for file watching and reading.
use std::fmt;

use paths::AbsPathBuf;

pub enum Entry {
    Files(Vec<AbsPathBuf>),
    Directory { path: AbsPathBuf, globs: Vec<String> },
}

pub struct Config {
    pub load: Vec<Entry>,
    pub watch: Vec<usize>,
}

pub enum Message {
    DidSwitchConfig { n_entries: usize },
    DidLoadAllEntries,
    Loaded { files: Vec<(AbsPathBuf, Option<Vec<u8>>)> },
}

pub type Sender = Box<dyn Fn(Message) + Send>;

pub trait Handle: fmt::Debug {
    fn spawn(sender: Sender) -> Self
    where
        Self: Sized;
    fn set_config(&mut self, config: Config);
    fn invalidate(&mut self, path: AbsPathBuf);
    fn load_sync(&mut self, path: &AbsPathBuf) -> Option<Vec<u8>>;
}

impl Entry {
    pub fn rs_files_recursively(base: AbsPathBuf) -> Entry {
        Entry::Directory { path: base, globs: globs(&["*.rs"]) }
    }
    pub fn local_cargo_package(base: AbsPathBuf) -> Entry {
        Entry::Directory { path: base, globs: globs(&["*.rs", "!/target/"]) }
    }
    pub fn cargo_package_dependency(base: AbsPathBuf) -> Entry {
        Entry::Directory {
            path: base,
            globs: globs(&["*.rs", "!/tests/", "!/examples/", "!/benches/"]),
        }
    }
}

fn globs(globs: &[&str]) -> Vec<String> {
    globs.iter().map(|it| it.to_string()).collect()
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::Loaded { files } => {
                f.debug_struct("Loaded").field("n_files", &files.len()).finish()
            }
            Message::DidSwitchConfig { n_entries } => {
                f.debug_struct("DidSwitchConfig").field("n_entries", n_entries).finish()
            }
            Message::DidLoadAllEntries => f.debug_struct("DidLoadAllEntries").finish(),
        }
    }
}

#[test]
fn handle_is_object_safe() {
    fn _assert(_: &dyn Handle) {}
}
