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
    /// | `UserDirs` | [`SearchPathDirectory`] |
    /// | ---------- | ----------------------- |
    /// | [`cache_home`] | [`.cachesDirectory`] (`~/Library/Caches`) |
    /// | [`config_home`] | [`.applicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`data_home`] | [`.applicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`state_home`] | [`.applicationSupportDirectory`] (`~/Library/Application Support`) |
    /// | [`desktop`] | [`.desktopDirectory`] (`~/Desktop`) |
    /// | [`documents`] | [`.documentDirectory`] (`~/Documents`) |
    /// | [`downloads`] | [`.downloadsDirectory`] |
    /// | [`music`] | [`.musicDirectory`] (`~/Music`) |
    /// | [`pictures`] | [`.picturesDirectory`] (`~/Pictures`) |
    /// | [`public_share`] | [`.sharedPublicDirectory`] (`~/Public`) |
    /// | [`videos`] | [`.moviesDirectory`] (`~/Movies`) |
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
    /// directory path is not valid Unicode, or if the [user home](home_dir)
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
    /// [`SearchPathDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory
    /// [`.cachesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/cachesdirectory
    /// [`.applicationSupportDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/applicationsupportdirectory
    /// [`.desktopDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/desktopdirectory
    /// [`.documentDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/documentdirectory
    /// [`.downloadsDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/downloadsdirectory
    /// [`.musicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/musicdirectory
    /// [`.picturesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/picturesdirectory
    /// [`.sharedPublicDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/sharedpublicdirectory
    /// [`.moviesDirectory`]: https://developer.apple.com/documentation/foundation/filemanager/searchpathdirectory/moviesdirectory
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn sysdir() -> io::Result<Self>;
}

#[unstable(feature = "dir_discovery", issue = "157515")]
#[cfg(target_vendor = "apple")]
impl UserDirsExt for UserDirs {
    fn sysdir() -> io::Result<Self> {
        let mut dirs = UserDirs::new();
        let home =
            dirs.user_home().ok_or(const_error!(ErrorKind::NotFound, "no home directory"))?;

        let caches = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_CACHES)?;
        let application_support =
            get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_APPLICATION_SUPPORT)?;
        let desktop = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_DESKTOP)?;
        let documents = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_DOCUMENT)?;
        let downloads = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_DOWNLOADS)?;
        let movies = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_MOVIES)?;
        let music = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_MUSIC)?;
        let pictures = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_PICTURES)?;
        let public_share = get_user_sysdir(&home, sys::SYSDIR_DIRECTORY_SHARED_PUBLIC)?;

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

/// Get the path for a system directory using `sysdir(3)`.
#[cfg(target_vendor = "apple")]
fn get_user_sysdir(
    home: &Path,
    kind: sys::sysdir_search_path_directory_t,
) -> io::Result<Option<PathBuf>> {
    // SAFETY: `kind` is a valid `sysdir_search_path_directory_t` value, and
    //   `SYSDIR_DOMAIN_MASK_USER` is a valid `sysdir_search_path_domain_mask_t` value.
    let state =
        unsafe { sys::sysdir_start_search_path_enumeration(kind, sys::SYSDIR_DOMAIN_MASK_USER) };
    if state == 0 {
        // 0 directory paths produced
        return Ok(None);
    }

    let mut path = [0u8; sys::PATH_MAX as usize];
    // SAFETY: `state` is nonzero and comes from `sysdir_start_search_path_enumeration`
    // SAFETY: sysdir_get_next_search_path_enumeration will write at most `PATH_MAX-1` bytes to `path`
    let state = unsafe {
        sys::sysdir_get_next_search_path_enumeration(state, path.as_mut_ptr() as *mut c_char)
    };
    if state == 0 {
        // exactly 1 directory path produced
        let Ok(path) = CStr::from_bytes_until_nul(&path) else {
            return Err(const_error!(
                ErrorKind::InvalidData,
                "standard user directory path too long",
            ));
        };
        // read as UTF-8
        let Ok(path) = path.to_str() else {
            return Err(const_error!(
                ErrorKind::InvalidData,
                "standard user directory path not valid UTF-8",
            ));
        };
        // expand the `~` shorthand
        return if path == "~" {
            Ok(Some(home.into()))
        } else if path.starts_with("~/") {
            Ok(Some(home.join(&path[2..])))
        } else if path.starts_with("~") {
            Err(const_error!(
                ErrorKind::InvalidData,
                "standard user directory relative to different user",
            ))
        } else if path.starts_with('/') {
            Ok(Some(path.into()))
        } else {
            Err(const_error!(ErrorKind::InvalidData, "standard user directory path not absolute"))
        };
    }

    // 2+ directory paths produced
    let mut state = state;
    // exhaust iterator for cleanup; ignore the paths that get returned
    while state != 0 {
        // SAFETY: `state` is nonzero and is from a previous call to sysdir_get_next_search_path_enumeration
        state = unsafe {
            sys::sysdir_get_next_search_path_enumeration(state, path.as_mut_ptr() as *mut c_char)
        };
    }

    Err(const_error!(ErrorKind::InvalidData, "multiple paths returned for standard user directory"))
}

#[cfg(target_vendor = "apple")]
mod sys {
    pub use libc::sysdir_search_path_directory_t::*;
    pub use libc::sysdir_search_path_domain_mask_t::*;
    pub use libc::{
        PATH_MAX, sysdir_get_next_search_path_enumeration, sysdir_search_path_directory_t,
        sysdir_start_search_path_enumeration,
    };
}

#[cfg(test)]
#[cfg(target_vendor = "apple")]
mod tests {
    use super::*;

    #[test]
    fn can_fetch_sysdir_paths() {
        let dirs = UserDirs::sysdir().unwrap();
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
