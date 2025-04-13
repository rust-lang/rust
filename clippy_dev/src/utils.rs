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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum UpdateMode {
    Check,
    Change,
}

pub(crate) fn exit_with_failure() {
    println!(
        "Not all lints defined properly. \
                 Please run `cargo dev update_lints` to make sure all lints are defined properly."
    );
    process::exit(1);
}

/// Replaces a region in a file delimited by two lines matching regexes.
///
/// `path` is the relative path to the file on which you want to perform the replacement.
///
/// See `replace_region_in_text` for documentation of the other options.
///
/// # Panics
///
/// Panics if the path could not read or then written
pub(crate) fn replace_region_in_file(
    update_mode: UpdateMode,
    path: &Path,
    start: &str,
    end: &str,
    write_replacement: impl FnMut(&mut String),
) {
    let contents = fs::read_to_string(path).unwrap_or_else(|e| panic!("Cannot read from `{}`: {e}", path.display()));
    let new_contents = match replace_region_in_text(&contents, start, end, write_replacement) {
        Ok(x) => x,
        Err(delim) => panic!("Couldn't find `{delim}` in file `{}`", path.display()),
    };

    match update_mode {
        UpdateMode::Check if contents != new_contents => exit_with_failure(),
        UpdateMode::Check => (),
        UpdateMode::Change => {
            if let Err(e) = fs::write(path, new_contents.as_bytes()) {
                panic!("Cannot write to `{}`: {e}", path.display());
            }
        },
    }
}

/// Replaces a region in a text delimited by two strings. Returns the new text if both delimiters
/// were found, or the missing delimiter if not.
pub(crate) fn replace_region_in_text<'a>(
    text: &str,
    start: &'a str,
    end: &'a str,
    mut write_replacement: impl FnMut(&mut String),
) -> Result<String, &'a str> {
    let (text_start, rest) = text.split_once(start).ok_or(start)?;
    let (_, text_end) = rest.split_once(end).ok_or(end)?;

    let mut res = String::with_capacity(text.len() + 4096);
    res.push_str(text_start);
    res.push_str(start);
    write_replacement(&mut res);
    res.push_str(end);
    res.push_str(text_end);

    Ok(res)
}
