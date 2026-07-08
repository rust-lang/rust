//! XDG (X Desktop Group) related functionality for Unix platforms.
//!
//! The [XDG Base Directory Specification][basedir] defines a set of base
//! directories, relative to which user-specific files should be looked for. The
//! functions in this module provide those directory paths as configured by
//! the environment.
//!
//! Note that the use of these functions is not enforced by the system, and as
//! such, not all programs will necessarily respect all details of the XDG path
//! environment. This is a set of guidelines, and each program is ultimately
//! responsible for defining where and how it both reads and writes files.
//!
//! Use of XDG paths can be generally considered the conventional expectation
//! on Linux-based systems. Other Unix-based systems may or may not play well
//! with the XDG conventions.
//!
//! Directories returned by this module are not guaranteed to exist yet. If the
//! directory does not exist, an application should attempt to create it with
//! [permissions mode][super::fs::PermissionsExt::from_mode] `0o700`.
//!
//! [basedir]: https://specifications.freedesktop.org/basedir/latest/
#![unstable(feature = "xdg_basedir", issue = "157515")]

use crate::env::{home_dir, split_paths, var_os};
use crate::ffi::{OsStr, OsString};
use crate::path::PathBuf;

fn xdg_home_dir() -> PathBuf {
    // Note: home_dir can return `Some("")` in some cases. We assume that in
    // this case the expected behavior is for `$HOME/path` to become `/path`,
    // i.e. the home directory is effectively `/`.
    match home_dir() {
        None => panic!("an XDG environment should have a home directory"),
        Some(home) if home.is_empty() => PathBuf::from("/"),
        Some(home) => home,
    }
}

fn xdg_dir(env: &str, fallback_home_subdir: &str) -> PathBuf {
    var_os(env)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| xdg_home_dir().join(fallback_home_subdir))
}

/// A base directory relative to which user-specific data files should be written.
///
/// An application `appid` would typically be expected to write its data files
/// to `{data_home_dir}/{appid}/**/*`.
pub fn data_home_dir() -> PathBuf {
    xdg_dir("XDG_DATA_HOME", ".local/share")
}

/// A base directory relative to which user-specific configuration files should be written.
///
/// An application `appid` would typically be expected to write its configuration
/// files to `{config_home_dir}/{appid}/**/*`.
pub fn config_home_dir() -> PathBuf {
    xdg_dir("XDG_CONFIG_HOME", ".config")
}

/// A base directory relative to which user-specific state data should be written.
///
/// An application `appid` would typically be expected to write its state data to
/// `{state_home_dir}/{appid}/**/*`.
///
/// Common kinds of state data include actions history (such as logs, history,
/// recently used files, etc.) and state of the application that can be reused
/// after application restart (such as view, layout, open files, undo history,
/// etc.).
pub fn state_home_dir() -> PathBuf {
    xdg_dir("XDG_STATE_HOME", ".local/state")
}

/// A base directory relative to which user-specific non-essential (cached) data should be written.
///
/// An application `appid` would typically be expected to write its cache data to
/// `{cache_home_dir}/{appid}/**/*`.
pub fn cache_home_dir() -> PathBuf {
    xdg_dir("XDG_CACHE_HOME", ".cache")
}

/// An iterator that produces directory paths from XDG environment configuration.
///
/// The iterator element type is [`PathBuf`].
///
/// This structure is created by [`xdg::data_dirs`] and [`xdg::config_dirs`].
/// See the documentation of those functions for more.
///
/// [`xdg::data_dirs`]: data_dirs
/// [`xdg::config_dirs`]: config_dirs
#[derive(Debug, Clone)]
pub struct XdgDirsIter {
    list: OsString,
    off: usize,
}

impl XdgDirsIter {
    fn new(env: &str, default: &str) -> Self {
        let dirs = var_os(env).filter(|s| !s.is_empty()).unwrap_or_else(|| default.into());
        Self { list: dirs, off: 0 }
    }

    fn remaining(&self) -> Option<&OsStr> {
        self.list.as_encoded_bytes().get(self.off..).map(|bytes| {
            // SAFETY: `self.off` is the index after a path separator (or the
            //     start of the string), so is a valid OsStr boundary.
            unsafe { OsStr::from_encoded_bytes_unchecked(bytes) }
        })
    }
}

impl Iterator for XdgDirsIter {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.remaining()?;
        let next = split_paths(rest).next()?;
        let len = next.as_os_str().len();
        self.off += len + 1; // Offset after this path and the separator after it.
        Some(next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let Some(dirs) = self.remaining() else { return (0, Some(0)) };
        split_paths(dirs).size_hint()
    }
}

/// A set of preference ordered directories relative to which data files should be searched.
///
/// If an application defines a data file to be at `$XDG_DATA_DIRS/appid/file.name`, this means that:
///
/// - The initial data file should be installed to `{system_data_dir}/appid/file.name`.
/// - A user-specific version of the data file may be created at
///   <code>{[data_home_dir][]()}/appid/file.name</code>.
/// - Lookups for the data file should search for `./appid/file.name` relative to
///   `data_home_dir` and each directory in `data_dirs`, giving preference to
///   files found relative to an earlier directory in the search order.
///
/// An application may choose to handle a file being located under multiple base
/// directories however it sees fit, so long as it respects the search order.
/// For example, it could say that only the first file found is used, or that
/// data within the files is merged in some way.
pub fn data_dirs() -> XdgDirsIter {
    // NB: the spec uses trailing slashes only for this default, for some reason
    XdgDirsIter::new("XDG_DATA_DIRS", "/usr/local/share/:/usr/share/")
}

/// A set of preference ordered directories relative to which configuration files should be searched.
///
/// If an application defines a configuration file to be at `$XDG_CONFIG_DIRS/appid/file.name`, this means that:
///
/// - The initial configuration file should be installed to `{system_config_dir}/xdg/appid/file.name`.
/// - A user-specific version of the configuration file may be created at
///   <code>{[config_home_dir][]()}/appid/file.name</code>.
/// - Lookups for the configuration file should search for `./appid/file.name`
///   relative to `config_home_dir` and each directory in `config_dirs`, giving
///   preference to files found relative to an earlier directory in the search order.
///
/// An application may choose to handle a file being located under multiple base
/// directories however it sees fit, so long as it respects the search order.
/// For example, it could say that only the first file found is used, or that
/// data within the files is merged in some way.
pub fn config_dirs() -> XdgDirsIter {
    XdgDirsIter::new("XDG_CONFIG_DIRS", "/etc/xdg")
}
