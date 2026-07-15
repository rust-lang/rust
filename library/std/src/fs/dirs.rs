use crate::mem;
use crate::path::{Path, PathBuf};
use crate::sys::fs::{ExtraHomeDirs, ExtraMediaDirs};

/// Common user directory paths used for user-specific application files.
///
/// It is not required that the user directories are accessible by the current
/// user, nor that there is a directory at that path. A robust application
/// should handle the case where user directories are incorrectly configured.
///
/// Even when configured correctly, multiple paths may be to the same location.
/// As such, you should not assume that a file written relative to one directory
/// will not conflict with the same relative path in a different home directory.
///
/// # Platform-specific behavior
///
/// As the filesystem conventions for discovering directories varies between
/// operating systems, constructors for `HomeDirs` that use the host platform's
/// filesystem conventions are provided as extension traits under the `std::os`
/// module.
#[unstable(feature = "dir_discovery", issue = "157515")]
#[derive(Debug, Clone)]
pub struct HomeDirs {
    pub(crate) cache: Option<PathBuf>,
    pub(crate) config: Option<PathBuf>,
    pub(crate) data: Option<PathBuf>,
    pub(crate) state: Option<PathBuf>,
    #[cfg_attr(not(unix), expect(dead_code, reason = "no extra home dirs"))]
    pub(crate) extra: ExtraHomeDirs,
}

/// Common user directory paths used for user-specific media files.
///
/// It is not required that the media directories are accessible by the current
/// user, nor that there is a directory at that path. A robust application
/// should handle the case where media directories are incorrectly configured.
///
/// Even when configured correctly, multiple paths may be to the same location.
/// As such, you should not assume that a file written relative to one directory
/// will not conflict with the same relative path in a different media directory.
///
/// # Platform-specific behavior
///
/// As the filesystem conventions for discovering directories varies between
/// operating systems, constructors for `MediaDirs` that use the host platform's
/// filesystem conventions are provided as extension traits under the `std::os`
/// module.
#[unstable(feature = "media_dir_discovery", issue = "157515")]
#[derive(Debug, Clone)]
pub struct MediaDirs {
    pub(crate) desktop: Option<PathBuf>,
    pub(crate) documents: Option<PathBuf>,
    pub(crate) downloads: Option<PathBuf>,
    pub(crate) music: Option<PathBuf>,
    pub(crate) pictures: Option<PathBuf>,
    pub(crate) videos: Option<PathBuf>,
    #[cfg_attr(not(unix), expect(dead_code, reason = "no extra media dirs"))]
    pub(crate) extra: ExtraMediaDirs,
}

impl HomeDirs {
    /// Create a known user directory set with no known directories.
    ///
    /// This is useful with the builder `set_*` methods to create a `HomeDirs`
    /// with exactly the directories you want, without any other defaults.
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn empty() -> Self {
        Self { cache: None, config: None, data: None, state: None, extra: Default::default() }
    }

    /// A base directory relative to which user-specific non-essential cache
    /// data files should be stored.
    ///
    /// Files in this directory may be automatically purged any time they are
    /// not currently open.
    ///
    /// This is the same directory for all applications. As such, applications
    /// should use a subdirectory for application-specific cache files.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `${XDG_CACHE_HOME:-$HOME/.cache}` |
    /// | [Darwin] (macOS) | [`NSCachesDirectory`] (`$HOME/Library/Caches`) |
    /// | [Windows] | [`{FOLDERID_LocalAppData}`] (`%LOCALAPPDATA%`) |
    ///
    /// Other paths can be configured via [`set_cache_home`](Self::set_cache_home).
    ///
    /// [XDG]: crate::os::unix::fs::HomeDirsExt
    /// [Darwin]: crate::os::darwin::fs::HomeDirsExt
    /// [Windows]: crate::os::windows::fs::HomeDirsExt
    /// [`NSCachesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/cachesdirectory?language=objc
    /// [`{FOLDERID_LocalAppData}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_localappdata
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn cache_home(&self) -> Option<&Path> {
        self.cache.as_deref()
    }

    /// A base directory relative to which user-specific configuration files
    /// should be stored.
    ///
    /// This is the same directory for all applications. As such, applications
    /// should use a subdirectory for application-specific configuration files.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `${XDG_CONFIG_HOME:-$HOME/.config}` |
    /// | [Darwin] (macOS) | [`NSApplicationSupportDirectory`] (`$HOME/Library/Application Support`) |
    /// | [Windows] | [`{FOLDERID_RoamingAppData}`] (`%APPDATA%`) |
    ///
    /// Other paths can be configured via [`set_config_home`](Self::set_config_home).
    ///
    /// [XDG]: crate::os::unix::fs::HomeDirsExt
    /// [Darwin]: crate::os::darwin::fs::HomeDirsExt
    /// [Windows]: crate::os::windows::fs::HomeDirsExt
    /// [`NSApplicationSupportDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/applicationsupportdirectory?language=objc
    /// [`{FOLDERID_RoamingAppData}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_roamingappdata
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn config_home(&self) -> Option<&Path> {
        self.config.as_deref()
    }

    /// A base directory relative to which user-specific data files should be
    /// stored.
    ///
    /// This is the same directory for all applications. As such, applications
    /// should use a subdirectory for application-specific data files.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `${XDG_DATA_HOME:-$HOME/.local/share}` |
    /// | [Darwin] (macOS) | [`NSApplicationSupportDirectory`] (`$HOME/Library/Application Support`) |
    /// | [Windows] | [`{FOLDERID_RoamingAppData}`] (`%APPDATA%`) |
    ///
    /// Other paths can be configured via [`set_data_home`](Self::set_data_home).
    ///
    /// [XDG]: crate::os::unix::fs::HomeDirsExt
    /// [Darwin]: crate::os::darwin::fs::HomeDirsExt
    /// [Windows]: crate::os::windows::fs::HomeDirsExt
    /// [`NSApplicationSupportDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/applicationsupportdirectory?language=objc
    /// [`{FOLDERID_RoamingAppData}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_roamingappdata
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn data_home(&self) -> Option<&Path> {
        self.data.as_deref()
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
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `${XDG_STATE_HOME:-$HOME/.local/state}` |
    /// | [Darwin] (macOS) | [`NSApplicationSupportDirectory`] (`$HOME/Library/Application Support`) |
    /// | [Windows] | [`{FOLDERID_LocalAppData}`] (`%LOCALAPPDATA%`) |
    ///
    /// Other paths can be configured via [`set_state_home`](Self::set_state_home).
    ///
    /// [XDG]: crate::os::unix::fs::HomeDirsExt
    /// [Darwin]: crate::os::darwin::fs::HomeDirsExt
    /// [Windows]: crate::os::windows::fs::HomeDirsExt
    /// [`NSApplicationSupportDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/applicationsupportdirectory?language=objc
    /// [`{FOLDERID_LocalAppData}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_localappdata
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn state_home(&self) -> Option<&Path> {
        self.state.as_deref()
    }
}

impl MediaDirs {
    /// Create a known user directory set with no known directories.
    ///
    /// This is useful with the builder `set_*` methods to create a `MediaDirs`
    /// with exactly the directories you want, without any other defaults.
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn empty() -> Self {
        Self {
            desktop: None,
            documents: None,
            downloads: None,
            music: None,
            pictures: None,
            videos: None,
            extra: Default::default(),
        }
    }

    /// The OS-privileged user "Desktop" directory, often the `Desktop`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `$XDG_DESKTOP_DIR` (`$HOME/Desktop`) |
    /// | [Darwin] (macOS) | [`NSDesktopDirectory`] (`$HOME/Desktop`) |
    /// | [Windows] | [`{FOLDERID_Desktop}`] (`%USERPROFILE%\Desktop`) |
    ///
    /// Other paths can be configured via [`set_desktop`](Self::set_desktop).
    ///
    /// [XDG]: crate::os::unix::fs::MediaDirsExt
    /// [Darwin]: crate::os::darwin::fs::MediaDirsExt
    /// [Windows]: crate::os::windows::fs::MediaDirsExt
    /// [`NSDesktopDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/desktopdirectory?language=objc
    /// [`{FOLDERID_Desktop}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_desktop
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn desktop(&self) -> Option<&Path> {
        self.desktop.as_deref()
    }

    /// The OS-privileged user "Documents" directory, often the `Documents`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `$XDG_DOCUMENTS_DIR` (`$HOME/Documents`) |
    /// | [Darwin] (macOS) | [`NSDocumentDirectory`] (`$HOME/Documents`) |
    /// | [Windows] | [`{FOLDERID_Documents}`] (`%USERPROFILE%\Documents`) |
    ///
    /// Other paths can be configured via [`set_documents`](Self::set_documents).
    ///
    /// [XDG]: crate::os::unix::fs::MediaDirsExt
    /// [Darwin]: crate::os::darwin::fs::MediaDirsExt
    /// [Windows]: crate::os::windows::fs::MediaDirsExt
    /// [`NSDocumentDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/documentdirectory?language=objc
    /// [`{FOLDERID_Documents}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_documents
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn documents(&self) -> Option<&Path> {
        self.documents.as_deref()
    }

    /// The OS-privileged user "Downloads" directory, often the `Downloads`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `$XDG_DOWNLOAD_DIR` (`$HOME/Downloads`) |
    /// | [Darwin] (macOS) | [`NSDownloadsDirectory`] (`$HOME/Downloads`) |
    /// | [Windows] | [`{FOLDERID_Downloads}`] (`%USERPROFILE%\Downloads`) |
    ///
    /// Other paths can be configured via [`set_downloads`](Self::set_downloads).
    ///
    /// [XDG]: crate::os::unix::fs::MediaDirsExt
    /// [Darwin]: crate::os::darwin::fs::MediaDirsExt
    /// [Windows]: crate::os::windows::fs::MediaDirsExt
    /// [`NSDownloadsDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/downloadsdirectory?language=objc
    /// [`{FOLDERID_Downloads}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_downloads
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn downloads(&self) -> Option<&Path> {
        self.downloads.as_deref()
    }

    /// The OS-privileged user "Music" directory, often the `Music`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `$XDG_MUSIC_DIR` (`$HOME/Music`) |
    /// | [Darwin] (macOS) | [`NSMusicDirectory`] (`$HOME/Music`) |
    /// | [Windows] | [`{FOLDERID_Music}`] (`%USERPROFILE%\Music`) |
    ///
    /// Other paths can be configured via [`set_music`](Self::set_music).
    ///
    /// [XDG]: crate::os::unix::fs::MediaDirsExt
    /// [Darwin]: crate::os::darwin::fs::MediaDirsExt
    /// [Windows]: crate::os::windows::fs::MediaDirsExt
    /// [`NSMusicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/musicdirectory?language=objc
    /// [`{FOLDERID_Music}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_music
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn music(&self) -> Option<&Path> {
        self.music.as_deref()
    }

    /// The OS-privileged user "Pictures" directory, often the `Pictures`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `$XDG_PICTURES_DIR` (`$HOME/Pictures`) |
    /// | [Darwin] (macOS) | [`NSPicturesDirectory`] (`$HOME/Pictures`) |
    /// | [Windows] | [`{FOLDERID_Pictures}`] (`%USERPROFILE%\Pictures`) |
    ///
    /// Other paths can be configured via [`set_pictures`](Self::set_pictures).
    ///
    /// [XDG]: crate::os::unix::fs::MediaDirsExt
    /// [Darwin]: crate::os::darwin::fs::MediaDirsExt
    /// [Windows]: crate::os::windows::fs::MediaDirsExt
    /// [`NSPicturesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/picturesdirectory?language=objc
    /// [`{FOLDERID_Pictures}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_pictures
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn pictures(&self) -> Option<&Path> {
        self.pictures.as_deref()
    }

    /// The OS-privileged user "Videos" directory, often the `Videos`
    /// folder in the user's home directory.
    ///
    /// As a media directory, this should typically be used as a default path
    /// for file selection dialogs, not for automatically accessed file paths.
    ///
    /// # Platform-specific behavior
    ///
    /// When constructed using platform-specific conventions, the value is:
    ///
    /// | OS | Path |
    /// | -- | ---- |
    /// | [XDG] (Linux) | `$XDG_VIDEOS_DIR` (`$HOME/Videos`) |
    /// | [Darwin] (macOS) | [`NSMoviesDirectory`] (`$HOME/Movies`) |
    /// | [Windows] | [`{FOLDERID_Videos}`] (`%USERPROFILE%\Videos`) |
    ///
    /// Other paths can be configured via [`set_videos`](Self::set_videos).
    ///
    /// [XDG]: crate::os::unix::fs::MediaDirsExt
    /// [Darwin]: crate::os::darwin::fs::MediaDirsExt
    /// [Windows]: crate::os::windows::fs::MediaDirsExt
    /// [`NSMoviesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/moviesdirectory?language=objc
    /// [`{FOLDERID_Videos}`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_videos
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn videos(&self) -> Option<&Path> {
        self.videos.as_deref()
    }
}

impl HomeDirs {
    /// Take the contents of this directory set, leaving it empty.
    ///
    /// This is useful with the builder `set_*` methods to build `HomeDirs`
    /// without needing an intermediate mutable binding.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(dir_discovery)]
    /// use std::fs::HomeDirs;
    ///
    /// # /*
    /// let base_dir = /* ... */;
    /// # */ let base_dir = std::path::PathBuf::from("/");
    /// let dirs = HomeDirs::empty()
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

    /// Set the path for [Self::cache_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_cache_home(&mut self, path: PathBuf) -> &mut Self {
        self.cache = Some(path);
        self
    }

    /// Set the path for [Self::config_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_config_home(&mut self, path: PathBuf) -> &mut Self {
        self.config = Some(path);
        self
    }

    /// Set the path for [Self::data_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_data_home(&mut self, path: PathBuf) -> &mut Self {
        self.data = Some(path);
        self
    }

    /// Set the path for [Self::state_home].
    #[unstable(feature = "dir_discovery", issue = "157515")]
    pub fn set_state_home(&mut self, path: PathBuf) -> &mut Self {
        self.state = Some(path);
        self
    }
}

impl MediaDirs {
    /// Take the contents of this directory set, leaving it empty.
    ///
    /// This is useful with the builder `set_*` methods to build `MediaDirs`
    /// without needing an intermediate mutable binding.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(media_dir_discovery)]
    /// use std::fs::MediaDirs;
    ///
    /// # /*
    /// let base_dir = /* ... */;
    /// # */ let base_dir = std::path::PathBuf::from("/");
    /// let dirs = MediaDirs::empty()
    ///    .set_desktop(base_dir.join("desktop"))
    ///    .set_documents(base_dir.join("documents"))
    ///    .set_downloads(base_dir.join("downloads"))
    ///    .set_music(base_dir.join("music"))
    ///    .set_pictures(base_dir.join("pictures"))
    ///    .set_videos(base_dir.join("videos"))
    ///    .take();
    /// ```
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn take(&mut self) -> Self {
        mem::replace(self, Self::empty())
    }

    /// Set the path for [Self::desktop].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_desktop(&mut self, path: PathBuf) -> &mut Self {
        self.desktop = Some(path);
        self
    }

    /// Set the path for [Self::documents].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_documents(&mut self, path: PathBuf) -> &mut Self {
        self.documents = Some(path);
        self
    }

    /// Set the path for [Self::downloads].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_downloads(&mut self, path: PathBuf) -> &mut Self {
        self.downloads = Some(path);
        self
    }

    /// Set the path for [Self::music].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_music(&mut self, path: PathBuf) -> &mut Self {
        self.music = Some(path);
        self
    }

    /// Set the path for [Self::pictures].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_pictures(&mut self, path: PathBuf) -> &mut Self {
        self.pictures = Some(path);
        self
    }

    /// Set the path for [Self::videos].
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    pub fn set_videos(&mut self, path: PathBuf) -> &mut Self {
        self.videos = Some(path);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_home_dirs_field_hookup_matches() {
        let mut dirs = HomeDirs::empty();

        assert_eq!(dirs.config_home(), None);
        assert_eq!(dirs.data_home(), None);
        assert_eq!(dirs.state_home(), None);
        assert_eq!(dirs.cache_home(), None);

        dirs.set_config_home("/config".into());
        dirs.set_data_home("/data".into());
        dirs.set_state_home("/state".into());
        dirs.set_cache_home("/cache".into());

        assert_eq!(dirs.config_home(), Some("/config".as_ref()));
        assert_eq!(dirs.data_home(), Some("/data".as_ref()));
        assert_eq!(dirs.state_home(), Some("/state".as_ref()));
        assert_eq!(dirs.cache_home(), Some("/cache".as_ref()));
    }

    #[test]
    fn test_media_dirs_field_hookup_matches() {
        let mut dirs = MediaDirs::empty();

        assert_eq!(dirs.desktop(), None);
        assert_eq!(dirs.documents(), None);
        assert_eq!(dirs.downloads(), None);
        assert_eq!(dirs.music(), None);
        assert_eq!(dirs.pictures(), None);
        assert_eq!(dirs.videos(), None);

        dirs.set_desktop("/desktop".into());
        dirs.set_documents("/documents".into());
        dirs.set_downloads("/downloads".into());
        dirs.set_music("/music".into());
        dirs.set_pictures("/pictures".into());
        dirs.set_videos("/videos".into());

        assert_eq!(dirs.desktop(), Some("/desktop".as_ref()));
        assert_eq!(dirs.documents(), Some("/documents".as_ref()));
        assert_eq!(dirs.downloads(), Some("/downloads".as_ref()));
        assert_eq!(dirs.music(), Some("/music".as_ref()));
        assert_eq!(dirs.pictures(), Some("/pictures".as_ref()));
        assert_eq!(dirs.videos(), Some("/videos".as_ref()));
    }
}
