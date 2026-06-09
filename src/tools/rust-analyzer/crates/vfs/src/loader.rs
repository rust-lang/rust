//! Dynamically compatible interface for file watching and reading.
use std::fmt;

use paths::{AbsPath, AbsPathBuf};

/// A set of files on the file system.
#[derive(Debug, Clone)]
pub enum Entry {
    /// The `Entry` is represented by a raw set of files.
    Files(Vec<AbsPathBuf>),
    /// The `Entry` is represented by `Directories`.
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
///
/// If a path is in both `include` and `exclude`, the `exclude` one wins.
#[derive(Debug, Clone, Default)]
pub struct Directories {
    pub extensions: Vec<String>,
    pub include: Vec<AbsPathBuf>,
    pub exclude: Vec<AbsPathBuf>,
}

/// [`Handle`]'s configuration.
#[derive(Debug)]
pub struct Config {
    /// Version number to associate progress updates to the right config
    /// version.
    pub version: u32,
    /// Set of initially loaded files.
    pub load: Vec<Entry>,
    /// Index of watched entries in `load`.
    ///
    /// If a path in a watched entry is modified,the [`Handle`] should notify it.
    pub watch: Vec<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LoadingProgress {
    Started,
    Progress(usize),
    Finished,
}

/// Message about an action taken by a [`Handle`].
pub enum Message {
    /// Indicate a gradual progress.
    ///
    /// This is supposed to be the number of loaded files.
    Progress {
        /// The total files to be loaded.
        n_total: usize,
        /// The files that have been loaded successfully.
        n_done: LoadingProgress,
        /// The dir being loaded, `None` if its for a file.
        dir: Option<AbsPathBuf>,
        /// The [`Config`] version.
        config_version: u32,
    },
    /// The handle loaded the following files' content for the first time.
    Loaded { files: Vec<(AbsPathBuf, Option<Vec<u8>>)> },
    /// The handle loaded the following files' content.
    Changed { files: Vec<(AbsPathBuf, Option<Vec<u8>>)> },
}

/// Type that will receive [`Messages`](Message) from a [`Handle`].
pub type Sender = crossbeam_channel::Sender<Message>;

/// Interface for reading and watching files.
pub trait Handle: fmt::Debug {
    /// Spawn a new handle with the given `sender`.
    fn spawn(sender: Sender) -> Self
    where
        Self: Sized;

    /// Set this handle's configuration.
    fn set_config(&mut self, config: Config);

    /// The file's content at `path` has been modified, and should be reloaded.
    fn invalidate(&mut self, path: AbsPathBuf);

    /// Load the content of the given file, returning [`None`] if it does not
    /// exists.
    fn load_sync(&mut self, path: &AbsPath) -> Option<Vec<u8>>;
}

impl Entry {
    /// Returns:
    /// ```text
    /// Entry::Directories(Directories {
    ///     extensions: ["rs"],
    ///     include: [base],
    ///     exclude: [base/.git],
    /// })
    /// ```
    pub fn rs_files_recursively(base: AbsPathBuf) -> Entry {
        Entry::Directories(dirs(base, &[".git"]))
    }

    /// Returns:
    /// ```text
    /// Entry::Directories(Directories {
    ///     extensions: ["rs"],
    ///     include: [base],
    ///     exclude: [base/.git, base/target],
    /// })
    /// ```
    pub fn local_cargo_package(base: AbsPathBuf) -> Entry {
        Entry::Directories(dirs(base, &[".git", "target"]))
    }

    /// Returns:
    /// ```text
    /// Entry::Directories(Directories {
    ///     extensions: ["rs"],
    ///     include: [base],
    ///     exclude: [base/.git, /tests, /examples, /benches],
    /// })
    /// ```
    pub fn cargo_package_dependency(base: AbsPathBuf) -> Entry {
        Entry::Directories(dirs(base, &[".git", "/tests", "/examples", "/benches"]))
    }

    /// Returns `true` if `path` is included in `self`.
    ///
    /// See [`Directories::contains_file`].
    pub fn contains_file(&self, path: &AbsPath) -> bool {
        match self {
            Entry::Files(files) => files.iter().any(|it| it == path),
            Entry::Directories(dirs) => dirs.contains_file(path),
        }
    }

    /// Returns `true` if `path` is included in `self`.
    ///
    /// - If `self` is `Entry::Files`, returns `false`
    /// - Else, see [`Directories::contains_dir`].
    pub fn contains_dir(&self, path: &AbsPath) -> bool {
        match self {
            Entry::Files(_) => false,
            Entry::Directories(dirs) => dirs.contains_dir(path),
        }
    }
}

impl Directories {
    /// Returns `true` if `path` is included in `self`.
    pub fn contains_file(&self, path: &AbsPath) -> bool {
        // First, check the file extension...
        let ext = path.extension().unwrap_or_default();
        if self.extensions.iter().all(|it| it.as_str() != ext) {
            return false;
        }

        // Then, check for path inclusion...
        self.includes_path(path)
    }

    /// Returns `true` if `path` is included in `self`.
    ///
    /// Since `path` is supposed to be a directory, this will not take extension
    /// into account.
    pub fn contains_dir(&self, path: &AbsPath) -> bool {
        self.includes_path(path)
    }

    /// Returns `true` if `path` is included in `self`.
    ///
    /// It is included if
    ///   - An element in `self.include` is a prefix of `path`.
    ///   - This path is longer than any element in `self.exclude` that is a prefix
    ///     of `path`. In case of equality, exclusion wins.
    fn includes_path(&self, path: &AbsPath) -> bool {
        let mut include: Option<&AbsPathBuf> = None;
        for incl in &self.include {
            if path.starts_with(incl) {
                include = Some(match include {
                    Some(prev) if prev.starts_with(incl) => prev,
                    _ => incl,
                });
            }
        }

        let include = match include {
            Some(it) => it,
            None => return false,
        };

        !self.exclude.iter().any(|excl| path.starts_with(excl) && excl.starts_with(include))
    }
}

/// Returns :
/// ```text
/// Directories {
///     extensions: ["rs"],
///     include: [base],
///     exclude: [base/<exclude>],
/// }
/// ```
fn dirs(base: AbsPathBuf, exclude: &[&str]) -> Directories {
    let exclude = exclude.iter().map(|it| base.join(it)).collect::<Vec<_>>();
    Directories { extensions: vec!["rs".to_owned()], include: vec![base], exclude }
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::Loaded { files } => {
                f.debug_struct("Loaded").field("n_files", &files.len()).finish()
            }
            Message::Changed { files } => {
                f.debug_struct("Changed").field("n_files", &files.len()).finish()
            }
            Message::Progress { n_total, n_done, dir, config_version } => f
                .debug_struct("Progress")
                .field("n_total", n_total)
                .field("n_done", n_done)
                .field("dir", dir)
                .field("config_version", config_version)
                .finish(),
        }
    }
}

#[test]
fn handle_is_dyn_compatible() {
    fn _assert(_: &dyn Handle) {}
}
