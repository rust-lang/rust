use crate::ffi::{CStr, c_char};
use crate::fs::UserDirs;
use crate::io::{self, ErrorKind, const_error};
use crate::path::{Path, PathBuf};

trait Sealed {}
impl Sealed for UserDirs {}

/// Darwin-specific extensions to [`fs::UserDirs`](UserDirs).
#[unstable(feature = "dir_discovery", issue = "157515")]
#[expect(private_bounds, reason = "sealed")]
pub trait UserDirsExt: Sized + Sealed {
    /// Load the standard user directory paths for the current user.
    ///
    /// On iOS, tvOS, watchOS, visionOS, and sandboxed macOS applications,
    /// [`cache_home`], [`config_home`], [`data_home`], and [`state_home`]
    /// are within the application's sandbox bundle. Outside the sandbox,
    /// these are subdirectories of the `~/Library` directory on macOS.
    ///
    /// The produced directory paths are not guaranteed to be the canonical
    /// paths to the directories; they are allowed to be sandbox-redirected
    /// paths as long as the user directory is accessible there.
    ///
    /// The loaded common directories are:
    ///
    /// | `UserDirs` | [`NSSearchPathDirectory`] |
    /// | ---------- | ----------------------- |
    /// | [`cache_home`] | [`NSCachesDirectory`] (`~/Library/Caches`) |
    /// | [`config_home`] | [`NSApplicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`data_home`] | [`NSApplicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`state_home`] | [`NSApplicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`desktop`] | [`NSDesktopDirectory`] (`~/Desktop`) |
    /// | [`documents`] | [`NSDocumentDirectory`] (`~/Documents`) |
    /// | [`downloads`] | [`NSDownloadsDirectory`] |
    /// | [`music`] | [`NSMusicDirectory`] (`~/Music`) |
    /// | [`pictures`] | [`NSPicturesDirectory`] (`~/Pictures`) |
    /// | [`public_share`] | [`NSSharedPublicDirectory`] (`~/Public`) |
    /// | [`videos`] | [`NSMoviesDirectory`] (`~/Movies`) |
    ///
    /// Note that the Application Support directory is used for the config,
    /// data, and state directories. It is always possible for multiple user
    /// directories to be configured to the same path, but this is the common
    /// configuration on Apple platforms, making it even more important to not
    /// assume files in different user directories cannot alias each other.
    ///
    /// # Errors
    ///
    /// Errors if the underlying system directory discovery API returns
    /// multiple directories for a queried standard directory, if a
    /// directory path is not valid Unicode, or if the [user home](UserDirs::user_home)
    /// cannot be determined.
    ///
    /// # Implementation-specific behavior
    ///
    /// Uses the `sysdir(3)` API from `libSystem` to discover the standard
    /// user directories. Iterates each of `SYSDIR_DIRECTORY_DOCUMENT`,
    /// `SYSDIR_DIRECTORY_DESKTOP`, `SYSDIR_DIRECTORY_CACHES`,
    /// `SYSDIR_DIRECTORY_APPLICATION_SUPPORT`, `SYSDIR_DIRECTORY_DOWNLOADS`,
    /// `SYSDIR_DIRECTORY_MOVIES`, `SYSDIR_DIRECTORY_MUSIC`,
    /// `SYSDIR_DIRECTORY_PICTURES`, `SYSDIR_DIRECTORY_SHARED_PUBLIC` once.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load additional common
    /// directories not currently in this list.
    ///
    /// [`cache_home`]: UserDirs::cache_home
    /// [`config_home`]: UserDirs::config_home
    /// [`data_home`]: UserDirs::data_home
    /// [`state_home`]: UserDirs::state_home
    /// [`desktop`]: UserDirs::desktop
    /// [`documents`]: UserDirs::documents
    /// [`downloads`]: UserDirs::downloads
    /// [`music`]: UserDirs::music
    /// [`pictures`]: UserDirs::pictures
    /// [`public_share`]: UserDirs::public_share
    /// [`videos`]: UserDirs::videos
    ///
    /// [`NSSearchPathDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory?language=objc
    /// [`NSCachesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/cachesdirectory?language=objc
    /// [`NSApplicationSupportDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/applicationsupportdirectory?language=objc
    /// [`NSDesktopDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/desktopdirectory?language=objc
    /// [`NSDocumentDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/documentdirectory?language=objc
    /// [`NSDownloadsDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/downloadsdirectory?language=objc
    /// [`NSMusicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/musicdirectory?language=objc
    /// [`NSPicturesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/picturesdirectory?language=objc
    /// [`NSSharedPublicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/sharedpublicdirectory?language=objc
    /// [`NSMoviesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/moviesdirectory?language=objc
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn sysdir() -> io::Result<Self>;
}

#[unstable(feature = "dir_discovery", issue = "157515")]
#[cfg(target_vendor = "apple")]
impl UserDirsExt for UserDirs {
    fn sysdir() -> io::Result<Self> {
        use libc::sysdir_search_path_directory_t::*;

        let mut dirs = UserDirs::new();
        let home =
            dirs.user_home().ok_or(const_error!(ErrorKind::NotFound, "no home directory"))?;

        let caches = sys::get_user_dir(&home, SYSDIR_DIRECTORY_CACHES)?;
        let application_support = sys::get_user_dir(&home, SYSDIR_DIRECTORY_APPLICATION_SUPPORT)?;
        let desktop = sys::get_user_dir(&home, SYSDIR_DIRECTORY_DESKTOP)?;
        let documents = sys::get_user_dir(&home, SYSDIR_DIRECTORY_DOCUMENT)?;
        let downloads = sys::get_user_dir(&home, SYSDIR_DIRECTORY_DOWNLOADS)?;
        let movies = sys::get_user_dir(&home, SYSDIR_DIRECTORY_MOVIES)?;
        let music = sys::get_user_dir(&home, SYSDIR_DIRECTORY_MUSIC)?;
        let pictures = sys::get_user_dir(&home, SYSDIR_DIRECTORY_PICTURES)?;
        let public_share = sys::get_user_dir(&home, SYSDIR_DIRECTORY_SHARED_PUBLIC)?;

        dirs.home.cache = caches;
        // Apple puts config/data/state all in Application Support
        dirs.home.config = application_support.clone();
        dirs.home.data = application_support.clone();
        dirs.home.state = application_support;

        dirs.media.desktop = desktop;
        dirs.media.documents = documents;
        dirs.media.downloads = downloads;
        dirs.media.music = music;
        dirs.media.pictures = pictures;
        dirs.media.public_share = public_share;
        dirs.media.videos = movies;

        Ok(dirs)
    }
}

/// Safer wrapper around the sysdir(3) API
#[cfg(target_vendor = "apple")]
mod sys {
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
                        buf.as_mut_ptr() as *mut libc::c_char,
                    )
                };
            }

            if self.state == 0 {
                // exhausted
                return None;
            }

            let Ok(path) = std::ffi::CStr::from_bytes_until_nul(&buf) else {
                // should be impossible given username length limits, but be defensive
                return Some(Err(const_error!(
                    ErrorKind::InvalidData,
                    "standard user directory path too long",
                )));
            };

            let Ok(path) = path.to_str() else {
                // FIXME: is this possible, and if so, how should it be handled?
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
        let dirs = UserDirs::sysdir().unwrap();
        assert!(dirs.user_home().is_some());
        assert!(dirs.cache_home().is_some());
        assert!(dirs.config_home().is_some());
        assert!(dirs.data_home().is_some());
        assert!(dirs.state_home().is_some());
        assert!(dirs.desktop().is_some());
        assert!(dirs.documents().is_some());
        assert!(dirs.downloads().is_some());
        assert!(dirs.music().is_some());
        assert!(dirs.pictures().is_some());
        assert!(dirs.public_share().is_some());
        assert!(dirs.videos().is_some());
    }
}
