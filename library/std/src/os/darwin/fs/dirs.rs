use crate::env;
use crate::fs::{HomeDirs, MediaDirs};
use crate::io::{self, ErrorKind, const_error};
use crate::path::PathBuf;

/// Darwin-specific extensions to [`fs::HomeDirs`](HomeDirs).
#[unstable(feature = "dir_discovery", issue = "157515")]
pub impl(self) trait HomeDirsExt: Sized {
    /// Load the standard user directory paths for the current user.
    ///
    /// On iOS, tvOS, watchOS, visionOS, and sandboxed macOS applications,
    /// these directories are within the application's sandbox bundle. Outside
    /// the sandbox, these are subdirectories of the `~/Library` directory on
    /// macOS.
    ///
    /// The produced directory paths are not guaranteed to be the canonical
    /// paths to the directories; they are allowed to be sandbox-redirected
    /// paths as long as the directory is accessible there.
    ///
    /// The loaded common directories are:
    ///
    /// | `HomeDirs` | [`NSSearchPathDirectory`] |
    /// | ---------- | ----------------------- |
    /// | [`cache_home`] | [`NSCachesDirectory`] (`~/Library/Caches`) |
    /// | [`config_home`] | [`NSApplicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`data_home`] | [`NSApplicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`state_home`] | [`NSApplicationSupportDirectory`] (`~/Library/Application Support`) |
    ///
    /// Note that the Application Support directory is used for the config,
    /// data, and state directories. It is always possible for multiple user
    /// directories to be configured to the same path, but this is the common
    /// configuration on Apple platforms, making it even more important to not
    /// assume files in different user directories cannot alias each other.
    ///
    /// # Errors
    ///
    /// Errors if the the [user home](env::home_dir) cannot be determined.
    //  Errors due to the underlying sysdir(3) API should never occur, as
    //  - the user domain only has one directory for each search path;
    //  - the user domain always returns subdirectory paths of `~`;
    //  - the username plus OS defined path segments cannot exceed PATH_MAX; and
    //  - the username and OS defined path segments are always valid UTF-8.
    ///
    /// # Implementation-specific behavior
    ///
    /// Uses the `sysdir(3)` API from `libSystem` to discover the standard
    /// user directories.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load additional common
    /// directories not currently in this list.
    ///
    /// [`cache_home`]: HomeDirs::cache_home
    /// [`config_home`]: HomeDirs::config_home
    /// [`data_home`]: HomeDirs::data_home
    /// [`state_home`]: HomeDirs::state_home
    ///
    /// [`NSSearchPathDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory?language=objc
    /// [`NSCachesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/cachesdirectory?language=objc
    /// [`NSApplicationSupportDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/applicationsupportdirectory?language=objc
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn sysdir() -> io::Result<Self>;
}

/// Darwin-specific extensions to [`fs::MediaDirs`](MediaDirs).
#[unstable(feature = "media_dir_discovery", issue = "157515")]
pub impl(self) trait MediaDirsExt: Sized {
    /// Load the standard user directory paths for the current user.
    ///
    /// The produced directory paths are not guaranteed to be the canonical
    /// paths to the directories; they are allowed to be sandbox-redirected
    /// paths as long as the directory is accessible there.
    ///
    /// The loaded common directories are:
    ///
    /// | `MediaDirs` | [`NSSearchPathDirectory`] |
    /// | ---------- | ----------------------- |
    /// | [`desktop`] | [`NSDesktopDirectory`] (`~/Desktop`) |
    /// | [`documents`] | [`NSDocumentDirectory`] (`~/Documents`) |
    /// | [`downloads`] | [`NSDownloadsDirectory`] |
    /// | [`music`] | [`NSMusicDirectory`] (`~/Music`) |
    /// | [`pictures`] | [`NSPicturesDirectory`] (`~/Pictures`) |
    /// | [`videos`] | [`NSMoviesDirectory`] (`~/Movies`) |
    ///
    /// # Errors
    ///
    /// Errors if the the [user home](env::home_dir) cannot be determined.
    //  Errors due to the underlying sysdir(3) API should never occur, as
    //  - the user domain only has one directory for each search path;
    //  - the user domain always returns subdirectory paths of `~`;
    //  - the username plus OS defined path segments cannot exceed PATH_MAX; and
    //  - the username and OS defined path segments are always valid UTF-8.
    ///
    /// # Implementation-specific behavior
    ///
    /// Uses the `sysdir(3)` API from `libSystem` to discover the standard
    /// user directories.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load additional common
    /// directories not currently in this list.
    ///
    /// [`desktop`]: MediaDirs::desktop
    /// [`documents`]: MediaDirs::documents
    /// [`downloads`]: MediaDirs::downloads
    /// [`music`]: MediaDirs::music
    /// [`pictures`]: MediaDirs::pictures
    /// [`videos`]: MediaDirs::videos
    ///
    /// [`NSSearchPathDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory?language=objc
    /// [`NSDesktopDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/desktopdirectory?language=objc
    /// [`NSDocumentDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/documentdirectory?language=objc
    /// [`NSDownloadsDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/downloadsdirectory?language=objc
    /// [`NSMusicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/musicdirectory?language=objc
    /// [`NSPicturesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/picturesdirectory?language=objc
    /// [`NSSharedPublicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/sharedpublicdirectory?language=objc
    /// [`NSMoviesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/moviesdirectory?language=objc
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn sysdir() -> io::Result<Self>;
}

fn user_home() -> io::Result<PathBuf> {
    env::home_dir()
        .filter(|p| !p.is_empty())
        .ok_or(const_error!(ErrorKind::NotFound, "no home directory"))
}

#[unstable(feature = "dir_discovery", issue = "157515")]
#[cfg(target_vendor = "apple")]
impl HomeDirsExt for HomeDirs {
    fn sysdir() -> io::Result<Self> {
        use libc::sysdir_search_path_directory_t::*;

        let mut dirs = HomeDirs::empty();
        let home = user_home()?;

        let caches = sys::get_user_dir(&home, SYSDIR_DIRECTORY_CACHES)?;
        let application_support = sys::get_user_dir(&home, SYSDIR_DIRECTORY_APPLICATION_SUPPORT)?;

        dirs.cache = caches;
        // Apple puts config/data/state all in Application Support
        dirs.config = application_support.clone();
        dirs.data = application_support.clone();
        dirs.state = application_support;

        Ok(dirs)
    }
}

#[unstable(feature = "media_dir_discovery", issue = "157515")]
#[cfg(target_vendor = "apple")]
impl MediaDirsExt for MediaDirs {
    fn sysdir() -> io::Result<Self> {
        use libc::sysdir_search_path_directory_t::*;

        let mut dirs = MediaDirs::empty();
        let home = user_home()?;

        let desktop = sys::get_user_dir(&home, SYSDIR_DIRECTORY_DESKTOP)?;
        let documents = sys::get_user_dir(&home, SYSDIR_DIRECTORY_DOCUMENT)?;
        let downloads = sys::get_user_dir(&home, SYSDIR_DIRECTORY_DOWNLOADS)?;
        let movies = sys::get_user_dir(&home, SYSDIR_DIRECTORY_MOVIES)?;
        let music = sys::get_user_dir(&home, SYSDIR_DIRECTORY_MUSIC)?;
        let pictures = sys::get_user_dir(&home, SYSDIR_DIRECTORY_PICTURES)?;

        dirs.desktop = desktop;
        dirs.documents = documents;
        dirs.downloads = downloads;
        dirs.music = music;
        dirs.pictures = pictures;
        dirs.videos = movies;

        Ok(dirs)
    }
}

/// Safer wrapper around the sysdir(3) API
#[cfg(target_vendor = "apple")]
mod sys {
    use crate::ffi::{CStr, c_char};
    use crate::io::{self, ErrorKind, const_error};
    use crate::path::{Path, PathBuf};

    /// Get the path for a system directory using `sysdir(3)`.
    pub fn get_user_dir(
        home: &Path,
        kind: libc::sysdir_search_path_directory_t,
    ) -> io::Result<Option<PathBuf>> {
        use libc::sysdir_search_path_domain_mask_t::SYSDIR_DOMAIN_MASK_USER;

        // SAFETY: SYSDIR_DOMAIN_MASK_USER < SYSDIR_DOMAIN_MASK_ALL
        let mut iter = unsafe { Iter::new(home, kind, SYSDIR_DOMAIN_MASK_USER) };
        let Some(path) = iter.next() else {
            return Ok(None);
        };

        if iter.next().is_some() {
            // more than one path returned (shouldn't happen for SYSDIR_DOMAIN_MASK_USER)
            return Err(const_error!(
                ErrorKind::InvalidData,
                "multiple paths returned for standard user directory",
            ));
        }

        Ok(Some(path?))
    }

    struct Iter<'a> {
        home: &'a Path,
        state: libc::sysdir_search_path_enumeration_state,
    }

    impl Drop for Iter<'_> {
        fn drop(&mut self) {
            for _ in self {}
        }
    }

    impl<'a> Iter<'a> {
        // SAFETY: `mask` must be <= `SYSDIR_DOMAIN_MASK_ALL`
        pub unsafe fn new(
            home: &'a Path,
            kind: libc::sysdir_search_path_directory_t,
            mask: libc::sysdir_search_path_domain_mask_t,
        ) -> Self {
            // SAFETY: forwarded to the caller
            let state = unsafe { libc::sysdir_start_search_path_enumeration(kind, mask) };
            Self { home, state }
        }
    }

    impl Iterator for Iter<'_> {
        type Item = io::Result<PathBuf>;

        fn next(&mut self) -> Option<io::Result<PathBuf>> {
            let mut buf = [0u8; libc::PATH_MAX as usize];
            if self.state != 0 {
                // SAFETY: `self.state` is nonzero and comes from prior sysdir_{start|get_next}_search_path_enumeration call
                // SAFETY: sysdir_get_next_search_path_enumeration will write at most `PATH_MAX` bytes to `path`
                self.state = unsafe {
                    libc::sysdir_get_next_search_path_enumeration(
                        self.state,
                        buf.as_mut_ptr() as *mut c_char,
                    )
                };
            }

            if self.state == 0 {
                // exhausted
                return None;
            }

            let Ok(path) = CStr::from_bytes_until_nul(&buf) else {
                // should be impossible given username length limits, but be defensive
                return Some(Err(const_error!(
                    ErrorKind::InvalidData,
                    "standard user directory path too long",
                )));
            };

            let Ok(path) = path.to_str() else {
                // should be impossible on a working system, but be defensive
                return Some(Err(const_error!(
                    ErrorKind::InvalidData,
                    "standard user directory path not valid UTF-8",
                )));
            };

            // expand `~` shorthand
            Some(match path {
                "~" => Ok(self.home.into()),
                _ if path.starts_with("~/") => Ok(self.home.join(&path[2..])),
                _ if path.starts_with("~") => Err(const_error!(
                    ErrorKind::InvalidData,
                    "standard user directory relative to different user",
                )),
                _ => {
                    let path = PathBuf::from(path);
                    if path.is_relative() {
                        Err(const_error!(
                            ErrorKind::InvalidData,
                            "standard user directory path not absolute",
                        ))
                    } else {
                        Ok(path)
                    }
                }
            })
        }
    }
}

#[cfg(test)]
#[cfg(target_vendor = "apple")]
mod tests {
    use super::*;

    #[test]
    fn can_fetch_sysdir_paths() {
        let dirs = HomeDirs::sysdir().unwrap();
        assert!(dirs.cache_home().is_some());
        assert!(dirs.config_home().is_some());
        assert!(dirs.data_home().is_some());
        assert!(dirs.state_home().is_some());

        let dirs = MediaDirs::sysdir().unwrap();
        assert!(dirs.desktop().is_some());
        assert!(dirs.documents().is_some());
        assert!(dirs.downloads().is_some());
        assert!(dirs.music().is_some());
        assert!(dirs.pictures().is_some());
        assert!(dirs.videos().is_some());
    }
}
