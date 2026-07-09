use crate::ffi::OsString;
use crate::path::{Path, PathBuf};
use crate::{env, mem};

/// A set of known user-specific directories.
///
/// Most operating systems provide some way to discover the paths to some set
/// of "well-known" directories that it is recommended for applications to use
/// for files accessed at runtime that are not part of the application itself.
/// `UserDirs` provides an OS-agnostic way to manipulate these directories.
///
/// A note of caution, however: multiple of these directory paths may be the
/// same as each other, so you should not assume that a file written relative
/// to the [config](Self::config_home) directory will not conflict with the same
/// relative path in the [data](Self::data_home) directory, for example. Aliasing
/// directories in this way is the common practice of some operating systems,
/// so it is important that a program still works correctly even if user paths
/// alias each other.
///
/// It is not required that the user directories are for the current user, nor
/// that there is a directory at that path. A robust application should handle
/// the case where user directories are incorrectly configured.
///
/// # Platform-specific behavior
///
/// As the filesystem conventions for user-specific directories varries between
/// operating systems, constructors for `UserDirs` that discover the host OS's
/// filesystem conventions are provided as extension traits under the `std::os`
/// module.
#[unstable(feature = "dir_discovery", issue = "157515")]
#[derive(Debug, Clone)]
pub struct UserDirs {
    pub(crate) home: UserHomeDirs,
    pub(crate) media: UserMediaDirs,
    #[allow(dead_code)] // only used for XDG
    pub(crate) search: UserSearchDirs,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct UserHomeDirs {
    pub(crate) user: Option<PathBuf>,
    pub(crate) cache: Option<PathBuf>,
    pub(crate) config: Option<PathBuf>,
    pub(crate) data: Option<PathBuf>,
    pub(crate) state: Option<PathBuf>,
    #[allow(dead_code)] // only used for XDG
    pub(crate) runtime: Option<PathBuf>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct UserMediaDirs {
    pub(crate) desktop: Option<PathBuf>,
    pub(crate) documents: Option<PathBuf>,
    pub(crate) downloads: Option<PathBuf>,
    pub(crate) music: Option<PathBuf>,
    pub(crate) pictures: Option<PathBuf>,
    pub(crate) public_share: Option<PathBuf>,
    pub(crate) videos: Option<PathBuf>,
    #[allow(dead_code)] // only used for XDG
    pub(crate) templates: Option<PathBuf>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct UserSearchDirs {
    #[allow(dead_code)] // only used for XDG
    pub(crate) config: Option<OsString>,
    #[allow(dead_code)] // only used for XDG
    pub(crate) data: Option<OsString>,
}

impl UserDirs {
    /// Create a known user directory set with only [`user_home`] set.
    ///
    /// Unlike [`env::home_dir`], this treats an empty home path as `None`.
    ///
    /// This is useful with the builder `set_*` methods to create a `UserDirs`
    /// with exactly the directories you want, without any other defaults.
    ///
    /// [`user_home`]: Self::user_home
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn new() -> Self {
        let mut dirs = Self::empty();
        dirs.home.user = env::home_dir().filter(|p| !p.is_empty());
        dirs
    }

    /// Create a known user directory set with no known directories.
    ///
    /// This is useful with the builder `set_*` methods to create a `UserDirs`
    /// with exactly the directories you want, without any other defaults.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn empty() -> Self {
        Self { home: Default::default(), media: Default::default(), search: Default::default() }
    }

    /// The path to the user's home directory.
    ///
    /// Applications putting files in the user's home directory without being
    /// directed to is generally discouraged. This should typically be used as
    /// a default path for file selection dialogs, not for automatically
    /// accessed file paths. Use one of [`config_home`], [`data_home`],
    /// [`state_home`], or [`cache_home`] instead as appropriate for the file.
    ///
    /// [`config_home`]: Self::config_home
    /// [`data_home`]: Self::data_home
    /// [`state_home`]: Self::state_home
    /// [`cache_home`]: Self::cache_home
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn user_home(&self) -> Option<&Path> {
        self.home.user.as_deref()
    }

    /// A base directory relative to which user-specific non-essential cache
    /// data files should be stored.
    ///
    /// Files in this directory may be automatically purged any time they are
    /// not currently open.
    ///
    /// This is the same directory for all applications. As such, applications
    /// should use a subdirectory for application-specific cache files.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn cache_home(&self) -> Option<&Path> {
        self.home.cache.as_deref()
    }

    /// A base directory relative to which user-specific configuration files
    /// should be stored.
    ///
    /// This is the same directory for all applications. As such, applications

    /// should use a subdirectory for application-specific configuration files.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn config_home(&self) -> Option<&Path> {
        self.home.config.as_deref()
    }

    /// A base directory relative to which user-specific data files should be
    /// stored.
    ///
    /// This is the same directory for all applications. As such, applications
    /// should use a subdirectory for application-specific data files.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn data_home(&self) -> Option<&Path> {
        self.home.data.as_deref()
    }

    /// A base directory relative to which user-specific state files should be
    /// stored.
    ///
    /// "State" files are data that should persist between application restarts,
    /// but which is not important nor portable enough to the user to be stored
    /// in the [data](Self::data_home) directory. Common examples include history
    /// (such as logs, recently used files, etc) and any current state of the
    /// application that should be reused (such as view, layout, open files,
    /// undo history, etc).
    ///
    /// This is the same directory for all applications. As such, applications
    /// should use a subdirectory for application-specific state files.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn state_home(&self) -> Option<&Path> {
        self.home.state.as_deref()
    }

    /// The OS-privileged user "Desktop" directory, often the `Desktop`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn desktop(&self) -> Option<&Path> {
        self.media.desktop.as_deref()
    }

    /// The OS-privileged user "Documents" directory, often the `Documents`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn documents(&self) -> Option<&Path> {
        self.media.documents.as_deref()
    }

    /// The OS-privileged user "Downloads" directory, often the `Downloads`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn downloads(&self) -> Option<&Path> {
        self.media.downloads.as_deref()
    }

    /// The OS-privileged user "Music" directory, often the `Music`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn music(&self) -> Option<&Path> {
        self.media.music.as_deref()
    }

    /// The OS-privileged user "Public Share" directory, often the `Public`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn public_share(&self) -> Option<&Path> {
        self.media.public_share.as_deref()
    }

    /// The OS-privileged user "Pictures" directory, often the `Pictures`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn pictures(&self) -> Option<&Path> {
        self.media.pictures.as_deref()
    }

    /// The OS-privileged user "Videos" directory, often the `Videos`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn videos(&self) -> Option<&Path> {
        self.media.videos.as_deref()
    }
}

impl UserDirs {
    /// Take the contents of this directory set, leaving it empty.
    ///
    /// This is useful with the builder `set_*` methods to build `UserDirs`
    /// without needing an intermediate mutable binding.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(dir_discovery)]
    /// use std::fs::UserDirs;
    ///
    /// # /*
    /// let base_dir = /* ... */;
    /// # */ let base_dir = std::path::PathBuf::from("/");
    /// let dirs = UserDirs::empty()
    ///    .set_config_home(base_dir.join("config"))
    ///    .set_data_home(base_dir.join("data"))
    ///    .set_state_home(base_dir.join("state"))
    ///    .set_cache_home(base_dir.join("cache"))
    ///    .take();
    /// ```
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn take(&mut self) -> Self {
        mem::replace(self, Self::empty())
    }

    /// Set the path for [Self::user_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_user_home(&mut self, path: PathBuf) -> &mut Self {
        self.home.user = Some(path);
        self
    }

    /// Set the path for [Self::cache_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_cache_home(&mut self, path: PathBuf) -> &mut Self {
        self.home.cache = Some(path);
        self
    }

    /// Set the path for [Self::config_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_config_home(&mut self, path: PathBuf) -> &mut Self {
        self.home.config = Some(path);
        self
    }

    /// Set the path for [Self::data_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_data_home(&mut self, path: PathBuf) -> &mut Self {
        self.home.data = Some(path);
        self
    }

    /// Set the path for [Self::state_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_state_home(&mut self, path: PathBuf) -> &mut Self {
        self.home.state = Some(path);
        self
    }

    /// Set the path for [Self::desktop].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_desktop(&mut self, path: PathBuf) -> &mut Self {
        self.media.desktop = Some(path);
        self
    }

    /// Set the path for [Self::documents].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_documents(&mut self, path: PathBuf) -> &mut Self {
        self.media.documents = Some(path);
        self
    }

    /// Set the path for [Self::downloads].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_downloads(&mut self, path: PathBuf) -> &mut Self {
        self.media.downloads = Some(path);
        self
    }

    /// Set the path for [Self::music].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_music(&mut self, path: PathBuf) -> &mut Self {
        self.media.music = Some(path);
        self
    }

    /// Set the path for [Self::pictures].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_pictures(&mut self, path: PathBuf) -> &mut Self {
        self.media.pictures = Some(path);
        self
    }

    /// Set the path for [Self::public_share].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_public_share(&mut self, path: PathBuf) -> &mut Self {
        self.media.public_share = Some(path);
        self
    }

    /// Set the path for [Self::videos].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_videos(&mut self, path: PathBuf) -> &mut Self {
        self.media.videos = Some(path);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_dirs_field_hookup_matches() {
        let mut dirs = UserDirs::empty();

        assert_eq!(dirs.config_home(), None);
        assert_eq!(dirs.data_home(), None);
        assert_eq!(dirs.state_home(), None);
        assert_eq!(dirs.cache_home(), None);

        assert_eq!(dirs.desktop(), None);
        assert_eq!(dirs.documents(), None);
        assert_eq!(dirs.downloads(), None);
        assert_eq!(dirs.music(), None);
        assert_eq!(dirs.pictures(), None);
        assert_eq!(dirs.public_share(), None);
        assert_eq!(dirs.videos(), None);

        dirs.set_config_home("/config".into());
        dirs.set_data_home("/data".into());
        dirs.set_state_home("/state".into());
        dirs.set_cache_home("/cache".into());

        dirs.set_desktop("/desktop".into());
        dirs.set_documents("/documents".into());
        dirs.set_downloads("/downloads".into());
        dirs.set_music("/music".into());
        dirs.set_pictures("/pictures".into());
        dirs.set_public_share("/public_share".into());
        dirs.set_videos("/videos".into());

        assert_eq!(dirs.config_home(), Some("/config".as_ref()));
        assert_eq!(dirs.data_home(), Some("/data".as_ref()));
        assert_eq!(dirs.state_home(), Some("/state".as_ref()));
        assert_eq!(dirs.cache_home(), Some("/cache".as_ref()));

        assert_eq!(dirs.desktop(), Some("/desktop".as_ref()));
        assert_eq!(dirs.documents(), Some("/documents".as_ref()));
        assert_eq!(dirs.downloads(), Some("/downloads".as_ref()));
        assert_eq!(dirs.music(), Some("/music".as_ref()));
        assert_eq!(dirs.pictures(), Some("/pictures".as_ref()));
        assert_eq!(dirs.public_share(), Some("/public_share".as_ref()));
        assert_eq!(dirs.videos(), Some("/videos".as_ref()));
    }
}
