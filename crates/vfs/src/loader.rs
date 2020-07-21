//! Object safe interface for file watching and reading.
use std::fmt;

use paths::{AbsPath, AbsPathBuf};

#[derive(Debug, Clone)]
pub enum Entry {
    Files(Vec<AbsPathBuf>),
    Directories(Directories),
}

/// Specifies a set of files on the file system.
///
/// A file is included if:
///   * it has included extension
///   * it is under an `include` path
///   * it is not under `exclude` path
///
/// If many include/exclude paths match, the longest one wins.
#[derive(Debug, Clone, Default)]
pub struct Directories {
    pub extensions: Vec<String>,
    pub include: Vec<AbsPathBuf>,
    pub exclude: Vec<AbsPathBuf>,
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
        Entry::Directories(dirs(base, &[".git"]))
    }
    pub fn local_cargo_package(base: AbsPathBuf) -> Entry {
        Entry::Directories(dirs(base, &[".git", "target"]))
    }
    pub fn cargo_package_dependency(base: AbsPathBuf) -> Entry {
        Entry::Directories(dirs(base, &[".git", "/tests", "/examples", "/benches"]))
    }

    pub fn contains_file(&self, path: &AbsPath) -> bool {
        match self {
            Entry::Files(files) => files.iter().any(|it| it == path),
            Entry::Directories(dirs) => dirs.contains_file(path),
        }
    }
    pub fn contains_dir(&self, path: &AbsPath) -> bool {
        match self {
            Entry::Files(_) => false,
            Entry::Directories(dirs) => dirs.contains_dir(path),
        }
    }
}

impl Directories {
    pub fn contains_file(&self, path: &AbsPath) -> bool {
        let ext = path.extension().unwrap_or_default();
        if self.extensions.iter().all(|it| it.as_str() != ext) {
            return false;
        }
        self.includes_path(path)
    }
    pub fn contains_dir(&self, path: &AbsPath) -> bool {
        self.includes_path(path)
    }
    fn includes_path(&self, path: &AbsPath) -> bool {
        let mut include: Option<&AbsPathBuf> = None;
        for incl in &self.include {
            if path.starts_with(incl) {
                include = Some(match include {
                    Some(prev) if prev.starts_with(incl) => prev,
                    _ => incl,
                })
            }
        }
        let include = match include {
            Some(it) => it,
            None => return false,
        };
        for excl in &self.exclude {
            if path.starts_with(excl) && excl.starts_with(include) {
                return false;
            }
        }
        true
    }
}

fn dirs(base: AbsPathBuf, exclude: &[&str]) -> Directories {
    let exclude = exclude.iter().map(|it| base.join(it)).collect::<Vec<_>>();
    Directories { extensions: vec!["rs".to_string()], include: vec![base], exclude }
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
