// use shlex::bytes::Shlex;

use crate::env::{JoinPathsError, SplitPaths, home_dir, join_paths, split_paths, var_os};
use crate::ffi::{OsStr, OsString};
use crate::fs::{self, UserDirs};
use crate::io::{self, ErrorKind, const_error};
use crate::os::unix::ffi::{OsStrExt, OsStringExt};
use crate::path::{Path, PathBuf};

trait Sealed {}
impl Sealed for UserDirs {}

/// XDG-specific extensions to [`fs::UserDirs`](UserDirs).
///
/// The XDG conventions are defined by the Freedesktop.org project in the
/// [XDG Base Directory Specification][xdg-basedir] and the [xdg-user-dirs]
/// tool. These conventions have been largely adopted by Linux distributions.
///
/// The XDG conventions are written to be usable on any Unix-like filesystem,
/// thus this extension being provided in `os::unix` rather than `os::linux`.
/// However, while some tooling does use XDG conventions on macOS, note that
/// macOS has its own separate conventions for user directories. Consider
/// carefully what conventions your users will expect your application to
/// follow along with any legacy path compatibility you might need to support.
///
/// [xdg-basedir]: https://specifications.freedesktop.org/basedir/
/// [xdg-user-dirs]: https://www.freedesktop.org/wiki/Software/xdg-user-dirs/
#[unstable(feature = "dir_discovery", issue = "157515")]
#[expect(private_bounds, reason = "sealed")]
pub trait UserDirsExt: Sized + Sealed {
    /// Load the user directory paths according to the
    /// [XDG Base Directory Specification][xdg-basedir].
    ///
    /// Sets [`cache_home`], [`config_home`], [`data_home`], and
    /// [`state_home`], as well as the XDG-specific [`runtime_home`],
    /// [`config_dirs`], and [`data_dirs`]. Each of these are set to the value
    /// of their corresponding `XDG_*` environment variable if it is set and
    /// non-empty, else to the default value defined by the specification.
    ///
    /// | Field | Environment Variable | Default Value |
    /// | ----- | -------------------- | ------------- |
    /// | [`cache_home`] | `XDG_CACHE_HOME` | `$HOME/.cache` |
    /// | [`config_home`] | `XDG_CONFIG_HOME` | `$HOME/.config` |
    /// | [`data_home`] | `XDG_DATA_HOME` | `$HOME/.local/share` |
    /// | [`state_home`] | `XDG_STATE_HOME` | `$HOME/.local/state` |
    /// | [`runtime_home`] | `XDG_RUNTIME_DIR` | (see method docs) |
    /// | [`config_dirs`] | `XDG_CONFIG_DIRS` | [`/etc/xdg`] |
    /// | [`data_dirs`] | `XDG_DATA_DIRS` | [`/usr/local/share/`, `/usr/share/`] |
    ///
    /// Note that `$HOME` here means [`env::home_dir`](home_dir), which uses
    /// `$HOME` if set and non-empty, but falls back to the system password
    /// database if it isn't set. A correctly configured XDG system will have
    /// `$HOME` set, but this fallback matches that common to both the shell
    /// and the `xdg-user-dirs` tool.
    ///
    /// # Errors
    ///
    /// Errors if the user's home directory cannot be determined.
    ///
    /// [xdg-basedir]: https://specifications.freedesktop.org/basedir/
    /// [`cache_home`]: UserDirs::cache_home
    /// [`config_home`]: UserDirs::config_home
    /// [`data_home`]: UserDirs::data_home
    /// [`state_home`]: UserDirs::state_home
    /// [`runtime_home`]: UserDirsExt::runtime_home
    /// [`config_dirs`]: UserDirsExt::config_dirs
    /// [`data_dirs`]: UserDirsExt::data_dirs
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn xdg_base() -> io::Result<Self>;

    /// Load the user directory paths according to the xdg-user-dirs tool.
    ///
    /// In addition to the base directories set by [`xdg_base`], this also
    /// reads the `$XDG_CONFIG_HOME/user-dirs.dirs` file as defined by the
    /// [xdg-user-dirs] tool to set [`desktop`], [`documents`], [`downloads`],
    /// [`music`], [`pictures`], [`public_share`], and [`videos`], as well as
    /// the XDG-specific [`templates`].
    ///
    /// `user-dirs.dirs` uses "a shell format, so it's easy to access from a
    /// shell script," and the way the [xdg-user-dirs] tool suggests loading
    /// the configuration is to `source` the file in a shell. Instead of using
    /// the shell (and potentially executing arbitrary code), we directly read
    /// and parse the file.
    ///
    /// Only lines in the `XDG_{NAME}_DIR={path}` format are processed; all
    /// other lines are ignored. `{NAME}` is a known directory name defined by
    /// the xdg-user-dirs tool, and `{path}` is a double-quoted shell-escaped
    /// path; any line that does not match this format is silently ignored.
    /// Additionally, we follow the documentation and only expand a leading
    /// `$HOME/` prefix in the path; other shell expansions may work in other
    /// tools, but will be unexpanded in the returned path here if present.
    /// Furthermore, paths which are neither rooted nor relative to `$HOME`
    /// are ignored, as they are not valid according to the specification.
    ///
    /// # Errors
    ///
    /// Errors if the user's home directory cannot be determined or if the
    /// `$XDG_CONFIG_HOME/user-dirs.dirs` file cannot be read.
    ///
    /// [xdg-user-dirs]: https://www.freedesktop.org/wiki/Software/xdg-user-dirs/
    /// [`desktop`]: UserDirs::desktop
    /// [`documents`]: UserDirs::documents
    /// [`downloads`]: UserDirs::downloads
    /// [`music`]: UserDirs::music
    /// [`pictures`]: UserDirs::pictures
    /// [`public_share`]: UserDirs::public_share
    /// [`videos`]: UserDirs::videos
    /// [`templates`]: UserDirsExt::templates
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn xdg_user() -> io::Result<Self>;

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
    /// files *in addition to* [`config_home`](UserDirs::config_home).
    ///
    /// The order of directories denotes their importance; the first directory
    /// is the most important. Information defined relative to the more
    /// important base directory takes precedent. [`config_home`] is not
    /// necessarily present in this list, and is considered more important
    /// than any base directory in this list.
    #[unstable(feature = "dir_search_discovery", issue = "157515")]
    fn config_dirs(&self) -> Option<SplitPaths<'_>>;

    /// A preference-ordered list of base directories to search for data
    /// files *in addition to* [`data_home`](UserDirs::data_home).
    ///
    /// The order of directories denotes their importance; the first directory
    /// is the most important. Information defined relative to the more
    /// important base directory takes precedent. [`data_home`] is not
    /// necessarily present in this list, and is considered more important
    /// than any base directory in this list.
    #[unstable(feature = "dir_search_discovery", issue = "157515")]
    fn data_dirs(&self) -> Option<SplitPaths<'_>>;

    /// The OS-privileged user "Templates" directory, often the `Templates`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn templates(&self) -> Option<&Path>;

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

    /// Set the paths for [Self::templates].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn set_templates(&mut self, path: PathBuf) -> &mut Self;
}

#[unstable(feature = "dir_discovery", issue = "157515")]
impl UserDirsExt for UserDirs {
    fn xdg_base() -> io::Result<Self> {
        let mut dirs = Self::new();
        let user_home = home_dir()
            .filter(|p| !p.is_empty())
            .ok_or(const_error!(ErrorKind::NotFound, "no home directory"))?;

        dirs.home.cache = Some(xdg_dir("XDG_CACHE_HOME", || user_home.join(".cache")));
        dirs.home.config = Some(xdg_dir("XDG_CONFIG_HOME", || user_home.join(".config")));
        dirs.home.data = Some(xdg_dir("XDG_DATA_HOME", || user_home.join(".local/share")));
        dirs.home.state = Some(xdg_dir("XDG_STATE_HOME", || user_home.join(".local/state")));
        dirs.home.runtime = var_os("XDG_RUNTIME_DIR").filter(|s| !s.is_empty()).map(PathBuf::from);

        dirs.search.config = Some(xdg_env("XDG_CONFIG_DIRS", "/etc/xdg"));
        dirs.search.data = Some(xdg_env("XDG_DATA_DIRS", "/usr/local/share/:/usr/share/"));

        Ok(dirs)
    }

    fn xdg_user() -> io::Result<Self> {
        let mut dirs = Self::xdg_base()?;

        let spec = match fs::read(dirs.config_home().unwrap().join("user-dirs.dirs")) {
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
            // expand the only allowed expansion: a leading `$HOME/` prefix, still quoted
            let buffer;
            const HOME_RELATIVE_PREFIX: &[u8] = b"\"$HOME/";
            let expanded = if val.starts_with(HOME_RELATIVE_PREFIX) {
                let joined = dirs
                    .user_home()
                    .unwrap()
                    .join(OsStr::from_bytes(&val[HOME_RELATIVE_PREFIX.len()..]));
                buffer = OsString::from_iter([OsStr::new("\""), joined.as_os_str()]);
                buffer.as_bytes()
            } else {
                val
            };

            // the path value is shell-escaped, so unescape it
            // FIXME: get shlex working as dep-of-std
            // let mut lex = Shlex::new(expanded);
            // let Some(path) = lex.next() else { continue };
            // let None = lex.next() else { continue };
            let path = &expanded[1..expanded.len() - 1]; // FIXME: placeholder quote strip

            // load the known user directories
            match var {
                b"XDG_DESKTOP_DIR" => dirs.media.desktop = Some(path_from_bytes(&path)),
                b"XDG_DOCUMENTS_DIR" => dirs.media.documents = Some(path_from_bytes(&path)),
                b"XDG_DOWNLOAD_DIR" => dirs.media.downloads = Some(path_from_bytes(&path)),
                b"XDG_MUSIC_DIR" => dirs.media.music = Some(path_from_bytes(&path)),
                b"XDG_PICTURES_DIR" => dirs.media.pictures = Some(path_from_bytes(&path)),
                b"XDG_PUBLICSHARE_DIR" => dirs.media.public_share = Some(path_from_bytes(&path)),
                b"XDG_VIDEOS_DIR" => dirs.media.videos = Some(path_from_bytes(&path)),
                b"XDG_TEMPLATES_DIR" => dirs.media.templates = Some(path_from_bytes(&path)),
                _ => {
                    // ignore unknown variable assignment, matching shell permissiveness
                }
            }
        }

        Ok(dirs)
    }

    fn runtime_home(&self) -> Option<&Path> {
        self.home.runtime.as_deref()
    }

    fn config_dirs(&self) -> Option<SplitPaths<'_>> {
        self.search.config.as_ref().map(|s| split_paths(s))
    }

    fn data_dirs(&self) -> Option<SplitPaths<'_>> {
        self.search.data.as_ref().map(|s| split_paths(s))
    }

    fn templates(&self) -> Option<&Path> {
        self.media.templates.as_deref()
    }

    fn set_runtime_home(&mut self, path: PathBuf) -> &mut Self {
        self.home.runtime = Some(path);
        self
    }

    fn set_config_dirs(
        &mut self,
        paths: impl IntoIterator<Item: AsRef<OsStr>>,
    ) -> Result<&mut Self, JoinPathsError> {
        self.search.config = Some(join_paths(paths)?);
        Ok(self)
    }

    fn set_data_dirs(
        &mut self,
        paths: impl IntoIterator<Item: AsRef<OsStr>>,
    ) -> Result<&mut Self, JoinPathsError> {
        self.search.data = Some(join_paths(paths)?);
        Ok(self)
    }

    fn set_templates(&mut self, path: PathBuf) -> &mut Self {
        self.media.templates = Some(path);
        self
    }
}

fn xdg_dir(env: &str, fallback: impl FnOnce() -> PathBuf) -> PathBuf {
    var_os(env).filter(|s| !s.is_empty()).map(PathBuf::from).unwrap_or_else(fallback)
}

fn xdg_env(env: &str, fallback: &str) -> OsString {
    var_os(env).filter(|s| !s.is_empty()).unwrap_or_else(|| OsString::from(fallback))
}

fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(OsString::from_vec(bytes.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_fetch_xdg_base_dirs() {
        let dirs = UserDirs::xdg_base().unwrap();

        assert!(dirs.cache_home().is_some());
        assert!(dirs.config_home().is_some());
        assert!(dirs.data_home().is_some());
        assert!(dirs.state_home().is_some());
        // dirs.runtime() may not exist
        assert!(dirs.config_dirs().is_some());
        assert!(dirs.data_dirs().is_some());

        assert!(dirs.desktop().is_none());
        assert!(dirs.documents().is_none());
        assert!(dirs.downloads().is_none());
        assert!(dirs.music().is_none());
        assert!(dirs.pictures().is_none());
        assert!(dirs.public_share().is_none());
        assert!(dirs.videos().is_none());
        assert!(dirs.templates().is_none());
    }

    #[test]
    fn can_fetch_xdg_user_dirs() {
        let dirs = UserDirs::xdg_user().unwrap();

        assert!(dirs.cache_home().is_some());
        assert!(dirs.config_home().is_some());
        assert!(dirs.data_home().is_some());
        assert!(dirs.state_home().is_some());
        // dirs.runtime() may not exist
        assert!(dirs.config_dirs().is_some());
        assert!(dirs.data_dirs().is_some());

        assert!(dirs.desktop().is_some());
        assert!(dirs.documents().is_some());
        assert!(dirs.downloads().is_some());
        assert!(dirs.music().is_some());
        assert!(dirs.pictures().is_some());
        assert!(dirs.public_share().is_some());
        assert!(dirs.videos().is_some());
        assert!(dirs.templates().is_some());
    }
}
