//! Object safe interface for file watching and reading.
use std::fmt;

use paths::{AbsPath, AbsPathBuf};

#[derive(Debug)]
pub enum Entry {
    Files(Vec<AbsPathBuf>),
    Directory { path: AbsPathBuf, include: Vec<String> },
}

#[derive(Debug)]
pub struct Config {
    pub load: Vec<Entry>,
    pub watch: Vec<usize>,
}

pub enum Message {
    Progress { n_total: usize, n_done: usize },
    Loaded { files: Vec<(AbsPathBuf, Option<Vec<u8>>)> },
}

pub type Sender = Box<dyn Fn(Message) + Send>;

pub trait Handle: fmt::Debug {
    fn spawn(sender: Sender) -> Self
    where
        Self: Sized;
    fn set_config(&mut self, config: Config);
    fn invalidate(&mut self, path: AbsPathBuf);
    fn load_sync(&mut self, path: &AbsPath) -> Option<Vec<u8>>;
}

impl Entry {
    pub fn rs_files_recursively(base: AbsPathBuf) -> Entry {
        Entry::Directory { path: base, include: globs(&["*.rs", "!/.git/"]) }
    }
    pub fn local_cargo_package(base: AbsPathBuf) -> Entry {
        Entry::Directory { path: base, include: globs(&["*.rs", "!/target/", "!/.git/"]) }
    }
    pub fn cargo_package_dependency(base: AbsPathBuf) -> Entry {
        Entry::Directory {
            path: base,
            include: globs(&["*.rs", "!/tests/", "!/examples/", "!/benches/", "!/.git/"]),
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
            Message::Progress { n_total, n_done } => f
                .debug_struct("Progress")
                .field("n_total", n_total)
                .field("n_done", n_done)
                .finish(),
        }
    }
}

#[test]
fn handle_is_object_safe() {
    fn _assert(_: &dyn Handle) {}
}
