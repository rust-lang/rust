use crate::borrow::Cow;
use crate::env::{self, JoinPathsError, SplitPaths, join_paths, split_paths, var_os};
use crate::ffi::{OsStr, OsString};
use crate::fs::{self, HomeDirs, MediaDirs};
use crate::io::{self, ErrorKind, const_error};
use crate::os::unix::ffi::{OsStrExt, OsStringExt};
use crate::path::{Path, PathBuf};

trait Sealed {}
impl Sealed for HomeDirs {}
impl Sealed for MediaDirs {}

/// XDG-specific extensions to [`fs::HomeDirs`](HomeDirs).
///
/// The XDG conventions are defined by the Freedesktop.org project in the
/// [XDG Base Directory Specification][xdg-basedir]. These conventions have
/// been largely adopted by Linux distributions.
///
/// The XDG conventions are written to be usable on any Unix-like filesystem,
/// thus this extension being provided in `os::unix` rather than `os::linux`.
/// However, while some tooling does use XDG conventions on macOS, note that
/// macOS has its own separate conventions for user directories. Consider
/// carefully what conventions your users will expect your application to
/// follow along with any legacy path compatibility you might need to support.
///
/// [xdg-basedir]: https://specifications.freedesktop.org/basedir/
#[unstable(feature = "dir_discovery", issue = "157515")]
#[expect(private_bounds, reason = "sealed")]
pub trait HomeDirsExt: Sized + Sealed {
    /// Load the user directory paths according to the
    /// [XDG Base Directory Specification][xdg-basedir].
    ///
    /// Each base directory path is set to the value of its corresponding
    /// `XDG_*` environment variable (if it is set and non-empty), else to
    /// the default value defined by the specification.
    ///
    /// | Field | Environment Variable | Default Value |
    /// | ----- | -------------------- | ------------- |
    /// | [`cache_home`] | `XDG_CACHE_HOME` | `$HOME/.cache` |
    /// | [`config_home`] | `XDG_CONFIG_HOME` | `$HOME/.config` |
    /// | [`data_home`] | `XDG_DATA_HOME` | `$HOME/.local/share` |
    /// | [`state_home`] | `XDG_STATE_HOME` | `$HOME/.local/state` |
    /// | [`runtime_home`] | `XDG_RUNTIME_DIR` | (see method docs) |
    /// | [`config_dirs`] | `XDG_CONFIG_DIRS` | `/etc/xdg` |
    /// | [`data_dirs`] | `XDG_DATA_DIRS` | `/usr/local/share/`, `/usr/share/` |
    ///
    /// Note that `$HOME` here means [`env::home_dir`], which uses
    /// `$HOME` if set and non-empty, but falls back to the system password
    /// database if it isn't set. A correctly configured XDG system will have
    /// `$HOME` set, but this fallback matches that of the shell.
    ///
    /// # Errors
    ///
    /// Errors if the user's home directory cannot be determined.
    ///
    /// [xdg-basedir]: https://specifications.freedesktop.org/basedir/
    /// [`cache_home`]: HomeDirs::cache_home
    /// [`config_home`]: HomeDirs::config_home
    /// [`data_home`]: HomeDirs::data_home
    /// [`state_home`]: HomeDirs::state_home
    /// [`runtime_home`]: HomeDirsExt::runtime_home
    /// [`config_dirs`]: HomeDirsExt::config_dirs
    /// [`data_dirs`]: HomeDirsExt::data_dirs
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn xdg() -> io::Result<Self>;

    /// A base directory relative to which user-specific runtime files
    /// (such as sockets, named pipes, etc) should be stored.
    ///
    /// Files in this directory may be subjected to periodic clean-up.
    /// Larger files should not be placed here, since it might reside in
    /// runtime memory and cannot necessarily be swapped out to disk.
    ///
    /// This path does not have a default if not set. If it isn't set,
    /// applications should fall back to a replacement directory with
    /// similar capabilities and print a warning message.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn runtime_home(&self) -> Option<&Path>;

    /// A preference-ordered list of base directories to search for config
    /// files *in addition to* [`config_home`].
    ///
    /// The order of directories denotes their importance; the first directory
    /// is the most important. Information defined relative to the more
    /// important base directory takes precedent. [`config_home`] is not
    /// necessarily present in this list, and is considered more important
    /// than any base directory in this list.
    ///
    /// [`config_home`]: HomeDirs::config_home
    #[unstable(feature = "dir_search_discovery", issue = "157515")]
    fn config_dirs(&self) -> Option<SplitPaths<'_>>;

    /// A preference-ordered list of base directories to search for data
    /// files *in addition to* [`data_home`].
    ///
    /// The order of directories denotes their importance; the first directory
    /// is the most important. Information defined relative to the more
    /// important base directory takes precedent. [`data_home`] is not
    /// necessarily present in this list, and is considered more important
    /// than any base directory in this list.
    ///
    /// [`data_home`]: HomeDirs::data_home
    #[unstable(feature = "dir_search_discovery", issue = "157515")]
    fn data_dirs(&self) -> Option<SplitPaths<'_>>;

    /// Set the paths for [Self::runtime_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn set_runtime_home(&mut self, path: PathBuf) -> &mut Self;

    /// Set the paths for [Self::config_dirs].
    #[unstable(feature = "dir_search_discovery", issue = "157515")]
    fn set_config_dirs(
        &mut self,
        paths: impl IntoIterator<Item: AsRef<OsStr>>,
    ) -> Result<&mut Self, JoinPathsError>;

    /// Set the paths for [Self::data_dirs].
    #[unstable(feature = "dir_search_discovery", issue = "157515")]
    fn set_data_dirs(
        &mut self,
        paths: impl IntoIterator<Item: AsRef<OsStr>>,
    ) -> Result<&mut Self, JoinPathsError>;
}

/// XDG-specific extensions to [`fs::MediaDirs`](MediaDirs).
///
/// The XDG conventions are defined by the Freedesktop.org project in the
/// [xdg-user-dirs]. This configuration is generally present on desktop Linux
/// distributions, although adoption is less widespread than the base directory
/// specification.
///
/// The XDG conventions are written to be usable on any Unix-like filesystem,
/// thus this extension being provided in `os::unix` rather than `os::linux`.
/// However, while some tooling does use XDG conventions on macOS, note that
/// macOS has its own separate conventions for user directories. Consider
/// carefully what conventions your users will expect your application to
/// follow along with any legacy path compatibility you might need to support.
///
/// [xdg-user-dirs]: https://www.freedesktop.org/wiki/Software/xdg-user-dirs/
#[unstable(feature = "media_dir_discovery", issue = "157515")]
#[expect(private_bounds, reason = "sealed")]
pub trait MediaDirsExt: Sized + Sealed {
    /// Load the user directory paths according to the [xdg-user-dirs] tool.
    ///
    /// This directly reads and parses the `$XDG_CONFIG_HOME/user-dirs.dirs`
    /// file as defined and maintained by the [xdg-user-dirs] tool.
    ///
    /// # Errors
    ///
    /// Errors if the user's home directory cannot be determined or if the
    /// `$XDG_CONFIG_HOME/user-dirs.dirs` file cannot be read.
    ///
    /// # Implementation-specific behavior
    ///
    /// Only the format maintained by xdg-user-dirs-update is supported. Any
    /// configuration that does not match the expected format will result in
    /// loading an unspecified path or `None` for that directory. To be more
    /// specific:
    ///
    /// - Any line not in the format of `XDG_{NAME}_DIR={path}` where `{NAME}`
    ///   is one of `DESKTOP`, `DOWNLOAD`, `TEMPLATES`, `PUBLICSHARE`,
    ///   `DOCUMENTS`, `MUSIC`, `PICTURES`, or `VIDEOS` is ignored.
    /// - `{path}` must be a `"`-quoted shell-escaped path.
    /// - `{path}` may only start with `/` or `$HOME/`. A home-relative path is
    ///   returned relative to [`env::home_dir`]; shell expansion is
    ///   not performed.
    /// - A directory set to just `$HOME` without a subdirectory is treated as
    ///   unsetting the directory, and results in a `None` value for that path.
    /// - When shell expansion syntax other than a leading `$HOME` is present,
    ///   no shell invocations will be done, but the produced directory path is
    ///   otherwise unspecified.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load paths in a more
    /// permissive manner, such as supporting more shell expansion syntax
    /// that xdg-user-dirs officially forbids putting in `user-dirs.dirs`
    /// but has de-facto support when `source`ing the file in a shell.
    ///
    /// [xdg-user-dirs]: https://www.freedesktop.org/wiki/Software/xdg-user-dirs/
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn xdg() -> io::Result<Self>;

    /// The OS-privileged user "Templates" directory, often the `Templates`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn templates(&self) -> Option<&Path>;

    /// Set the paths for [Self::templates].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn set_templates(&mut self, path: PathBuf) -> &mut Self;
}

#[unstable(feature = "dir_discovery", issue = "157515")]
impl HomeDirsExt for HomeDirs {
    fn xdg() -> io::Result<Self> {
        let mut dirs = HomeDirs::empty();
        let user_home = user_home()?;

        dirs.cache = Some(xdg_dir("XDG_CACHE_HOME", || user_home.join(".cache")));
        dirs.config = Some(xdg_dir("XDG_CONFIG_HOME", || user_home.join(".config")));
        dirs.data = Some(xdg_dir("XDG_DATA_HOME", || user_home.join(".local/share")));
        dirs.state = Some(xdg_dir("XDG_STATE_HOME", || user_home.join(".local/state")));
        dirs.extra.runtime = var_os("XDG_RUNTIME_DIR").filter(|s| !s.is_empty()).map(PathBuf::from);

        dirs.extra.config_path = Some(xdg_env("XDG_CONFIG_DIRS", "/etc/xdg"));
        dirs.extra.data_path = Some(xdg_env("XDG_DATA_DIRS", "/usr/local/share/:/usr/share/"));

        Ok(dirs)
    }

    fn runtime_home(&self) -> Option<&Path> {
        self.extra.runtime.as_deref()
    }

    fn config_dirs(&self) -> Option<SplitPaths<'_>> {
        self.extra.config_path.as_ref().map(|s| split_paths(s))
    }

    fn data_dirs(&self) -> Option<SplitPaths<'_>> {
        self.extra.data_path.as_ref().map(|s| split_paths(s))
    }

    fn set_runtime_home(&mut self, path: PathBuf) -> &mut Self {
        self.extra.runtime = Some(path);
        self
    }

    fn set_config_dirs(
        &mut self,
        paths: impl IntoIterator<Item: AsRef<OsStr>>,
    ) -> Result<&mut Self, JoinPathsError> {
        self.extra.config_path = Some(join_paths(paths)?);
        Ok(self)
    }

    fn set_data_dirs(
        &mut self,
        paths: impl IntoIterator<Item: AsRef<OsStr>>,
    ) -> Result<&mut Self, JoinPathsError> {
        self.extra.data_path = Some(join_paths(paths)?);
        Ok(self)
    }
}

#[unstable(feature = "media_dir_discovery", issue = "157515")]
impl MediaDirsExt for MediaDirs {
    fn xdg() -> io::Result<Self> {
        let mut dirs = MediaDirs::empty();
        let user_home = user_home()?;
        let config_home = xdg_dir("XDG_CONFIG_HOME", || user_home.join(".config"));

        let spec = match fs::read(config_home.join("user-dirs.dirs")) {
            Ok(spec) => spec,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                return Err(const_error!(
                    ErrorKind::NotFound,
                    "missing `$XDG_CONFIG_HOME/user-dirs.dirs`",
                ));
            }
            Err(e) => return Err(e),
        };

        for line in spec.split(|&b| b == b'\n') {
            // trim leading whitespace
            let trimmed = line.trim_ascii_start();
            // skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with(&[b'#']) {
                continue;
            }

            // only variable assignment lines are allowed; split on `=`
            let mut split = trimmed.splitn(2, |&b| b == b'=');
            // extract assignment parts; ignore lines not in this format
            let Some(var) = split.next() else { continue };
            let Some(val) = split.next() else { continue };
            debug_assert_eq!(split.next(), None);

            // ensure the path is absolute or $HOME-relative
            if !val.starts_with(b"\"/") && !val.starts_with(b"\"$HOME/") {
                continue;
            }

            // the path value is quoted; unquote it
            let Some(path) = xdg_unquote(val) else { continue };

            // expand the $HOME prefix if present
            let path = if path.starts_with(b"$HOME/") {
                user_home.join(OsStr::from_bytes(&path[6..]))
            } else {
                PathBuf::from(OsString::from_vec(path.into_owned()))
            };

            // load the known user directories
            match var {
                b"XDG_DESKTOP_DIR" => dirs.desktop = Some(path),
                b"XDG_DOCUMENTS_DIR" => dirs.documents = Some(path),
                b"XDG_DOWNLOAD_DIR" => dirs.downloads = Some(path),
                b"XDG_MUSIC_DIR" => dirs.music = Some(path),
                b"XDG_PICTURES_DIR" => dirs.pictures = Some(path),
                b"XDG_VIDEOS_DIR" => dirs.videos = Some(path),
                b"XDG_TEMPLATES_DIR" => dirs.extra.templates = Some(path),
                _ => {
                    // ignore unknown variable assignment
                }
            }
        }

        Ok(dirs)
    }

    fn templates(&self) -> Option<&Path> {
        self.extra.templates.as_deref()
    }

    fn set_templates(&mut self, path: PathBuf) -> &mut Self {
        self.extra.templates = Some(path);
        self
    }
}

fn user_home() -> io::Result<PathBuf> {
    env::home_dir()
        .filter(|p| !p.is_empty())
        .ok_or(const_error!(ErrorKind::NotFound, "no home directory"))
}

fn xdg_dir(env: &str, fallback: impl FnOnce() -> PathBuf) -> PathBuf {
    var_os(env).filter(|s| !s.is_empty()).map(PathBuf::from).unwrap_or_else(fallback)
}

fn xdg_env(env: &str, fallback: &str) -> OsString {
    var_os(env).filter(|s| !s.is_empty()).unwrap_or_else(|| OsString::from(fallback))
}

fn xdg_unquote(bytes: &[u8]) -> Option<Cow<'_, [u8]>> {
    let [b'"', bytes @ .., b'"'] = bytes else { return None };

    if !bytes.contains(&b'\\') {
        return Some(Cow::Borrowed(bytes));
    }

    let mut rest = bytes;
    let mut s = Vec::with_capacity(rest.len());
    loop {
        let i = rest
            .iter()
            .position(|&b| matches!(b, b'"' | b'\\' | b'$' | b'`'))
            .unwrap_or(rest.len());
        s.extend_from_slice(&rest[..i]);
        match &rest[i..] {
            [] => break,
            [b'\\', c @ (b'"' | b'\\' | b'$' | b'`'), tail @ ..] => {
                s.push(*c);
                rest = tail;
            }
            [b'\\', b'\n', tail @ ..] => {
                // line continuation
                rest = tail;
            }
            [b'\\', tail @ ..] => {
                s.push(b'\\');
                rest = tail;
            }
            [b'"' | b'$' | b'`', ..] => {
                // unsupported shell syntax
                return None;
            }
            _ => unreachable!(),
        }
    }

    Some(Cow::Owned(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_fetch_xdg_base_dirs() {
        let dirs = HomeDirs::xdg().unwrap();

        assert!(dirs.cache_home().is_some());
        assert!(dirs.config_home().is_some());
        assert!(dirs.data_home().is_some());
        assert!(dirs.state_home().is_some());
        // dirs.runtime() may not exist
        assert!(dirs.config_dirs().is_some());
        assert!(dirs.data_dirs().is_some());
    }

    #[test]
    #[cfg_attr(target_vendor = "apple", ignore = "Apple OSes don't use xdg-user-dirs")]
    fn can_fetch_xdg_media_dirs() {
        let dirs = match MediaDirs::xdg() {
            Ok(dirs) => dirs,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                // xdg-user-dirs not initialized on this system, skip the test
                return;
            }
            Err(e) => panic!("failed to fetch xdg user dirs: {e:?}"),
        };

        assert!(dirs.desktop().is_some());
        assert!(dirs.documents().is_some());
        assert!(dirs.downloads().is_some());
        assert!(dirs.music().is_some());
        assert!(dirs.pictures().is_some());
        assert!(dirs.videos().is_some());
        assert!(dirs.templates().is_some());
    }
}
