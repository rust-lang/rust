use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use core::fmt::{self, Display};
use core::str::FromStr;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Read as _, Seek as _, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{self, ExitStatus};

#[cfg(not(windows))]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy";
#[cfg(windows)]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy.exe";

#[cold]
#[track_caller]
fn panic_io(e: &io::Error, action: &str, path: &Path) -> ! {
    panic!("error {action} `{}`: {}", path.display(), *e)
}

/// Wrapper around `std::fs::File` which panics with a path on failure.
pub struct File<'a> {
    pub inner: fs::File,
    pub path: &'a Path,
}
impl<'a> File<'a> {
    /// Opens a file panicking on failure.
    #[track_caller]
    pub fn open(path: &'a (impl AsRef<Path> + ?Sized), options: &mut OpenOptions) -> Self {
        let path = path.as_ref();
        match options.open(path) {
            Ok(inner) => Self { inner, path },
            Err(e) => panic_io(&e, "opening", path),
        }
    }

    /// Opens a file if it exists, panicking on any other failure.
    #[track_caller]
    pub fn open_if_exists(path: &'a (impl AsRef<Path> + ?Sized), options: &mut OpenOptions) -> Option<Self> {
        let path = path.as_ref();
        match options.open(path) {
            Ok(inner) => Some(Self { inner, path }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => panic_io(&e, "opening", path),
        }
    }

    /// Opens and reads a file into a string, panicking of failure.
    #[track_caller]
    pub fn open_read_to_cleared_string<'dst>(
        path: &'a (impl AsRef<Path> + ?Sized),
        dst: &'dst mut String,
    ) -> &'dst mut String {
        Self::open(path, OpenOptions::new().read(true)).read_to_cleared_string(dst)
    }

    /// Read the entire contents of a file to the given buffer.
    #[track_caller]
    pub fn read_append_to_string<'dst>(&mut self, dst: &'dst mut String) -> &'dst mut String {
        match self.inner.read_to_string(dst) {
            Ok(_) => {},
            Err(e) => panic_io(&e, "reading", self.path),
        }
        dst
    }

    #[track_caller]
    pub fn read_to_cleared_string<'dst>(&mut self, dst: &'dst mut String) -> &'dst mut String {
        dst.clear();
        self.read_append_to_string(dst)
    }

    /// Replaces the entire contents of a file.
    #[track_caller]
    pub fn replace_contents(&mut self, data: &[u8]) {
        let res = match self.inner.seek(SeekFrom::Start(0)) {
            Ok(_) => match self.inner.write_all(data) {
                Ok(()) => self.inner.set_len(data.len() as u64),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        if let Err(e) = res {
            panic_io(&e, "writing", self.path);
        }
    }
}

/// Returns the path to the `cargo-clippy` binary
///
/// # Panics
///
/// Panics if the path of current executable could not be retrieved.
#[must_use]
pub fn cargo_clippy_path() -> PathBuf {
    let mut path = env::current_exe().expect("failed to get current executable name");
    path.set_file_name(CARGO_CLIPPY_EXE);
    path
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
}
impl FromStr for Version {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(s) = s.strip_prefix("0.")
            && let Some((major, minor)) = s.split_once('.')
            && let Ok(major) = major.parse()
            && let Ok(minor) = minor.parse()
        {
            Ok(Self { major, minor })
        } else {
            Err(())
        }
    }
}
impl Version {
    /// Displays the version as a rust version. i.e. `x.y.0`
    #[must_use]
    pub fn rust_display(self) -> impl Display {
        struct X(Version);
        impl Display for X {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}.{}.0", self.0.major, self.0.minor)
            }
        }
        X(self)
    }

    /// Displays the version as it should appear in clippy's toml files. i.e. `0.x.y`
    #[must_use]
    pub fn toml_display(self) -> impl Display {
        struct X(Version);
        impl Display for X {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0.{}.{}", self.0.major, self.0.minor)
            }
        }
        X(self)
    }
}

pub struct ClippyInfo {
    pub path: PathBuf,
    pub version: Version,
}
impl ClippyInfo {
    #[must_use]
    pub fn search_for_manifest() -> Self {
        let mut path = env::current_dir().expect("error reading the working directory");
        let mut buf = String::new();
        loop {
            path.push("Cargo.toml");
            if let Some(mut file) = File::open_if_exists(&path, OpenOptions::new().read(true)) {
                let mut in_package = false;
                let mut is_clippy = false;
                let mut version: Option<Version> = None;

                // Ad-hoc parsing to avoid dependencies. We control all the file so this
                // isn't actually a problem
                for line in file.read_to_cleared_string(&mut buf).lines() {
                    if line.starts_with('[') {
                        in_package = line.starts_with("[package]");
                    } else if in_package && let Some((name, value)) = line.split_once('=') {
                        match name.trim() {
                            "name" => is_clippy = value.trim() == "\"clippy\"",
                            "version"
                                if let Some(value) = value.trim().strip_prefix('"')
                                    && let Some(value) = value.strip_suffix('"') =>
                            {
                                version = value.parse().ok();
                            },
                            _ => {},
                        }
                    }
                }

                if is_clippy {
                    let Some(version) = version else {
                        panic!("error reading clippy version from {}", file.path.display());
                    };
                    path.pop();
                    return ClippyInfo { path, version };
                }
            }

            path.pop();
            assert!(
                path.pop(),
                "error finding project root, please run from inside the clippy directory"
            );
        }
    }
}

/// # Panics
/// Panics if given command result was failed.
pub fn exit_if_err(status: io::Result<ExitStatus>) {
    match status.expect("failed to run command").code() {
        Some(0) => {},
        Some(n) => process::exit(n),
        None => {
            eprintln!("Killed by signal");
            process::exit(1);
        },
    }
}

#[derive(Clone, Copy)]
pub enum UpdateStatus {
    Unchanged,
    Changed,
}
impl UpdateStatus {
    #[must_use]
    pub fn from_changed(value: bool) -> Self {
        if value { Self::Changed } else { Self::Unchanged }
    }

    #[must_use]
    pub fn is_changed(self) -> bool {
        matches!(self, Self::Changed)
    }
}

#[derive(Clone, Copy)]
pub enum UpdateMode {
    Change,
    Check,
}
impl UpdateMode {
    #[must_use]
    pub fn from_check(check: bool) -> Self {
        if check { Self::Check } else { Self::Change }
    }
}

#[derive(Default)]
pub struct FileUpdater {
    src_buf: String,
    dst_buf: String,
}
impl FileUpdater {
    fn update_file_checked_inner(
        &mut self,
        tool: &str,
        mode: UpdateMode,
        path: &Path,
        update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
    ) {
        let mut file = File::open(path, OpenOptions::new().read(true).write(true));
        file.read_to_cleared_string(&mut self.src_buf);
        self.dst_buf.clear();
        match (mode, update(path, &self.src_buf, &mut self.dst_buf)) {
            (UpdateMode::Check, UpdateStatus::Changed) => {
                eprintln!(
                    "the contents of `{}` are out of date\nplease run `{tool}` to update",
                    path.display()
                );
                process::exit(1);
            },
            (UpdateMode::Change, UpdateStatus::Changed) => file.replace_contents(self.dst_buf.as_bytes()),
            (UpdateMode::Check | UpdateMode::Change, UpdateStatus::Unchanged) => {},
        }
    }

    fn update_file_inner(&mut self, path: &Path, update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus) {
        let mut file = File::open(path, OpenOptions::new().read(true).write(true));
        file.read_to_cleared_string(&mut self.src_buf);
        self.dst_buf.clear();
        if update(path, &self.src_buf, &mut self.dst_buf).is_changed() {
            file.replace_contents(self.dst_buf.as_bytes());
        }
    }

    pub fn update_file_checked(
        &mut self,
        tool: &str,
        mode: UpdateMode,
        path: impl AsRef<Path>,
        update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
    ) {
        self.update_file_checked_inner(tool, mode, path.as_ref(), update);
    }

    #[expect(clippy::type_complexity)]
    pub fn update_files_checked(
        &mut self,
        tool: &str,
        mode: UpdateMode,
        files: &mut [(
            impl AsRef<Path>,
            &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
        )],
    ) {
        for (path, update) in files {
            self.update_file_checked_inner(tool, mode, path.as_ref(), update);
        }
    }

    pub fn update_file(
        &mut self,
        path: impl AsRef<Path>,
        update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
    ) {
        self.update_file_inner(path.as_ref(), update);
    }
}

/// Replaces a region in a text delimited by two strings. Returns the new text if both delimiters
/// were found, or the missing delimiter if not.
pub fn update_text_region(
    path: &Path,
    start: &str,
    end: &str,
    src: &str,
    dst: &mut String,
    insert: &mut impl FnMut(&mut String),
) -> UpdateStatus {
    let Some((src_start, src_end)) = src.split_once(start) else {
        panic!("`{}` does not contain `{start}`", path.display());
    };
    let Some((replaced_text, src_end)) = src_end.split_once(end) else {
        panic!("`{}` does not contain `{end}`", path.display());
    };
    dst.push_str(src_start);
    dst.push_str(start);
    let new_start = dst.len();
    insert(dst);
    let changed = dst[new_start..] != *replaced_text;
    dst.push_str(end);
    dst.push_str(src_end);
    UpdateStatus::from_changed(changed)
}

pub fn update_text_region_fn(
    start: &str,
    end: &str,
    mut insert: impl FnMut(&mut String),
) -> impl FnMut(&Path, &str, &mut String) -> UpdateStatus {
    move |path, src, dst| update_text_region(path, start, end, src, dst, &mut insert)
}

#[must_use]
pub fn is_ident_char(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
}

pub struct StringReplacer<'a> {
    searcher: AhoCorasick,
    replacements: &'a [(&'a str, &'a str)],
}
impl<'a> StringReplacer<'a> {
    #[must_use]
    pub fn new(replacements: &'a [(&'a str, &'a str)]) -> Self {
        Self {
            searcher: AhoCorasickBuilder::new()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(replacements.iter().map(|&(x, _)| x))
                .unwrap(),
            replacements,
        }
    }

    /// Replace substrings if they aren't bordered by identifier characters.
    pub fn replace_ident_fn(&self) -> impl Fn(&Path, &str, &mut String) -> UpdateStatus {
        move |_, src, dst| {
            let mut pos = 0;
            let mut changed = false;
            for m in self.searcher.find_iter(src) {
                if !is_ident_char(src.as_bytes().get(m.start().wrapping_sub(1)).copied().unwrap_or(0))
                    && !is_ident_char(src.as_bytes().get(m.end()).copied().unwrap_or(0))
                {
                    changed = true;
                    dst.push_str(&src[pos..m.start()]);
                    dst.push_str(self.replacements[m.pattern()].1);
                    pos = m.end();
                }
            }
            dst.push_str(&src[pos..]);
            UpdateStatus::from_changed(changed)
        }
    }
}

#[expect(clippy::must_use_candidate)]
pub fn try_rename_file(old_name: &Path, new_name: &Path) -> bool {
    match OpenOptions::new().create_new(true).write(true).open(new_name) {
        Ok(file) => drop(file),
        Err(e) if matches!(e.kind(), io::ErrorKind::AlreadyExists | io::ErrorKind::NotFound) => return false,
        Err(e) => panic_io(&e, "creating", new_name),
    }
    match fs::rename(old_name, new_name) {
        Ok(()) => true,
        Err(e) => {
            drop(fs::remove_file(new_name));
            if e.kind() == io::ErrorKind::NotFound {
                false
            } else {
                panic_io(&e, "renaming", old_name);
            }
        },
    }
}

#[must_use]
pub fn insert_at_marker(text: &str, marker: &str, new_text: &str) -> Option<String> {
    let i = text.find(marker)?;
    let (pre, post) = text.split_at(i);
    Some([pre, new_text, post].into_iter().collect())
}

pub fn rewrite_file(path: &Path, f: impl FnOnce(&str) -> Option<String>) {
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .open(path)
        .unwrap_or_else(|e| panic_io(&e, "opening", path));
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .unwrap_or_else(|e| panic_io(&e, "reading", path));
    if let Some(new_contents) = f(&buf) {
        file.rewind().unwrap_or_else(|e| panic_io(&e, "writing", path));
        file.write_all(new_contents.as_bytes())
            .unwrap_or_else(|e| panic_io(&e, "writing", path));
        file.set_len(new_contents.len() as u64)
            .unwrap_or_else(|e| panic_io(&e, "writing", path));
    }
}

pub fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap_or_else(|e| panic_io(&e, "writing", path));
}
