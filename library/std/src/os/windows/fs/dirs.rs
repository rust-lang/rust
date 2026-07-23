use crate::fs::{HomeDirs, MediaDirs};
use crate::path::PathBuf;
use crate::{env, io};

/// Windows-specific extensions to [`fs::HomeDirs`](HomeDirs).
#[unstable(feature = "dir_discovery", issue = "157515")]
pub impl(self) trait HomeDirsExt: Sized {
    /// Load the known user folder paths from environment variables.
    ///
    /// The loaded known folders are:
    ///
    /// | `HomeDirs` | Environment Variable |
    /// | ---------- | -------------------- |
    /// | [`cache_home`] | `%LOCALAPPDATA%` (`%USERPROFILE%\AppData\Local`) |
    /// | [`config_home`] | `%APPDATA%` (`%USERPROFILE%\AppData\Roaming`) |
    /// | [`data_home`] | `%APPDATA%` (`%USERPROFILE%\AppData\Roaming`) |
    /// | [`state_home`] | `%LOCALAPPDATA%` (`%USERPROFILE%\AppData\Local`) |
    ///
    /// Note that caches/state are both put in `AppData\Local`, and config/data
    /// in `AppData\Roaming`. It is always possible for multiple user directories
    /// to be configured to the same path, but this is the common configuration
    /// on Windows platforms, making it even more important to not assume files
    /// in different user directories cannot alias each other.
    ///
    /// # Errors
    ///
    /// Errors if `%APPDATA%` or `%LOCALAPPDATA%` are not set to absolute paths.
    ///
    /// # Implementation-specific behavior
    ///
    /// Windows keeps these environment variables updated to contain the paths
    /// to the configured folder path, but it is possible for the environment
    /// variables to not match the underlying system, such as when the user or
    /// a program modifies the environment directly, or if the configuration
    /// changed after the environment block was copied from the system.
    ///
    /// Unlike [`known_folders`](Self::known_folders), this does not require
    /// `Shell32.dll` and thus does not require the overhead of linking in
    /// DLLs that may result in Windows considering the application as a
    /// graphical application.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load additional common
    /// directories not currently in this list. The lack of configuration
    /// for a folder not currently in this list will not be an error and
    /// will result in a `None` value for that path in the returned value.
    ///
    /// [`cache_home`]: HomeDirs::cache_home
    /// [`config_home`]: HomeDirs::config_home
    /// [`data_home`]: HomeDirs::data_home
    /// [`state_home`]: HomeDirs::state_home
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn appdata_env() -> io::Result<Self>;

    /// Load the known user folder paths using the [Known Folders] API.
    ///
    /// The loaded known folders are:
    ///
    /// | `HomeDirs` | [`KNOWNFOLDERID`] |
    /// | ---------- | ----------------- |
    /// | [`cache_home`] | [`FOLDERID_LocalAppData`] (`%LOCALAPPDATA%`) |
    /// | [`config_home`] | [`FOLDERID_RoamingAppData`] (`%APPDATA%`) |
    /// | [`data_home`] | [`FOLDERID_RoamingAppData`] (`%APPDATA%`) |
    /// | [`state_home`] | [`FOLDERID_LocalAppData`] (`%LOCALAPPDATA%`) |
    ///
    /// Note that caches/state are both put in LocalAppData, and config/data
    /// in RoamingAppData. It is always possible for multiple user directories
    /// to be configured to the same path, but this is the common configuration
    /// on Windows platforms, making it even more important to not assume files
    /// in different user directories cannot alias each other.
    ///
    /// # Errors
    ///
    /// Errors if the underlying system discovery API returns an error.
    /// The lack of a configured path is not considered an error and results
    /// in a `None` value for that path in the returned `HomeDirs`.
    ///
    /// # Implementation-specific behavior
    ///
    /// Calls [`SHGetKnownFolderPath`] from `Shell32.dll` for the current user
    /// once for each known folder. Does not create the folder if missing.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load additional common
    /// directories not currently in this list.
    ///
    /// [Known Folders]: https://learn.microsoft.com/en-us/windows/win32/shell/known-folders
    /// [`SHGetKnownFolderPath`]: https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderpath
    /// [`KNOWNFOLDERID`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid
    /// [`FOLDERID_LocalAppData`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_localappdata
    /// [`FOLDERID_RoamingAppData`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_roamingappdata
    /// [`cache_home`]: HomeDirs::cache_home
    /// [`config_home`]: HomeDirs::config_home
    /// [`data_home`]: HomeDirs::data_home
    /// [`state_home`]: HomeDirs::state_home
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn known_folders() -> io::Result<Self>;
}

/// Windows-specific extensions to [`fs::MediaDirs`](MediaDirs).
#[unstable(feature = "media_dir_discovery", issue = "157515")]
pub impl(self) trait MediaDirsExt: Sized {
    /// Load the known user folder paths using the [Known Folders] API.
    ///
    /// The loaded known folders are:
    ///
    /// | `MediaDirs` | [`KNOWNFOLDERID`] |
    /// | ---------- | ----------------- |
    /// | [`desktop`] | [`FOLDERID_Desktop`] (`%USERPROFILE%\Desktop`) |
    /// | [`documents`] | [`FOLDERID_Documents`] (`%USERPROFILE%\Documents`) |
    /// | [`downloads`] | [`FOLDERID_Downloads`] (`%USERPROFILE%\Downloads`) |
    /// | [`music`] | [`FOLDERID_Music`] (`%USERPROFILE%\Music`) |
    /// | [`pictures`] | [`FOLDERID_Pictures`] (`%USERPROFILE%\Pictures`) |
    /// | [`videos`] | [`FOLDERID_Videos`] (`%USERPROFILE%\Videos`) |
    ///
    /// # Errors
    ///
    /// Errors if the underlying system discovery API returns an error.
    /// The lack of a configured path is not considered an error and results
    /// in a `None` value for that path in the returned `MediaDirs`.
    ///
    /// # Implementation-specific behavior
    ///
    /// Calls [`SHGetKnownFolderPath`] for the current user once for each known
    /// folder. Does not create the folder if missing.
    ///
    /// This behavior may change in the future. One example change that we
    /// explicitly reserve the right to make is to load additional common
    /// directories not currently in this list.
    ///
    /// [Known Folders]: https://learn.microsoft.com/en-us/windows/win32/shell/known-folders
    /// [`SHGetKnownFolderPath`]: https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderpath
    /// [`KNOWNFOLDERID`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid
    /// [`FOLDERID_Desktop`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_desktop
    /// [`FOLDERID_Documents`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_documents
    /// [`FOLDERID_Downloads`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_downloads
    /// [`FOLDERID_Music`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_music
    /// [`FOLDERID_Pictures`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_pictures
    /// [`FOLDERID_Videos`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_videos
    /// [`desktop`]: MediaDirs::desktop
    /// [`documents`]: MediaDirs::documents
    /// [`downloads`]: MediaDirs::downloads
    /// [`music`]: MediaDirs::music
    /// [`pictures`]: MediaDirs::pictures
    /// [`videos`]: MediaDirs::videos
    #[unstable(feature = "media_dir_discovery", issue = "157515")]
    fn known_folders() -> io::Result<Self>;
}

#[cfg(windows)]
#[unstable(feature = "dir_discovery", issue = "157515")]
impl HomeDirsExt for HomeDirs {
    fn appdata_env() -> io::Result<Self> {
        let roaming_app_data = env::var_os("APPDATA")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .ok_or(const_error!(io::ErrorKind::InvalidData, "non-absolute %APPDATA%"))?;
        let local_app_data = env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .ok_or(const_error!(io::ErrorKind::InvalidData, "non-absolute %LOCALAPPDATA%"))?;

        // AppData/Local -- system-local, doesn't make sense to sync to another
        // AppData/Roaming -- data that makes sense to sync across machines

        let mut dirs = HomeDirs::empty();

        dirs.cache = Some(local_app_data.clone());
        dirs.config = Some(roaming_app_data.clone());
        dirs.data = Some(roaming_app_data);
        dirs.state = Some(local_app_data);

        Ok(dirs)
    }

    fn known_folders() -> io::Result<Self> {
        use crate::sys::c;

        let local_app_data = sys::get_known_folder_path(&c::FOLDERID_LocalAppData)?;
        let roaming_app_data = sys::get_known_folder_path(&c::FOLDERID_RoamingAppData)?;

        // AppData/Local -- system-local, doesn't make sense to sync to another
        // AppData/Roaming -- data that makes sense to sync across machines

        let mut dirs = HomeDirs::empty();

        dirs.cache = local_app_data.clone();
        dirs.config = roaming_app_data.clone();
        dirs.data = roaming_app_data;
        dirs.state = local_app_data;

        Ok(dirs)
    }
}

#[cfg(windows)]
#[unstable(feature = "media_dir_discovery", issue = "157515")]
impl MediaDirsExt for MediaDirs {
    fn known_folders() -> io::Result<Self> {
        use crate::sys::c;

        let desktop = sys::get_known_folder_path(&c::FOLDERID_Desktop)?;
        let documents = sys::get_known_folder_path(&c::FOLDERID_Documents)?;
        let downloads = sys::get_known_folder_path(&c::FOLDERID_Downloads)?;
        let music = sys::get_known_folder_path(&c::FOLDERID_Music)?;
        let pictures = sys::get_known_folder_path(&c::FOLDERID_Pictures)?;
        let videos = sys::get_known_folder_path(&c::FOLDERID_Videos)?;

        // AppData/Local -- system-local, doesn't make sense to sync to another
        // AppData/Roaming -- data that makes sense to sync across machines

        let mut dirs = MediaDirs::empty();

        dirs.desktop = desktop;
        dirs.documents = documents;
        dirs.downloads = downloads;
        dirs.music = music;
        dirs.pictures = pictures;
        dirs.videos = videos;

        Ok(dirs)
    }
}

#[cfg(windows)]
mod sys {
    use crate::io::{self, ErrorKind, const_error};
    use crate::path::PathBuf;
    use crate::sys::{c, os2path};
    use crate::{ptr, slice};

    /// Retrieve a known folder path from the Windows API.
    pub fn get_known_folder_path(id: &c::GUID) -> io::Result<Option<PathBuf>> {
        // Get the known folder path. hToken = NULL requests the current user
        // scope, and we set KF_FLAG_DONT_VERIFY because it's a bit faster and
        // we don't guarantee that the directories at the paths exist.
        let mut pszPath = ptr::null_mut();
        // SAFETY: rfid/ppszPath are valid pointers, flags are appropriate, and
        //   a NULL hToken is supported by SHGetKnownFolderPath.
        let hr = unsafe {
            c::SHGetKnownFolderPath(
                /* rfid */ id,
                /* dwFlags */ c::KF_FLAG_DONT_VERIFY as _,
                /* hToken */ ptr::null_mut(),
                /* ppszPath */ &mut pszPath,
            )
        };

        let result = match hr {
            c::S_OK => {
                // SAFETY: pszPath was populated by a successful call to SHGetKnownFolderPath
                //   and is valid up to and including its nul terminator
                let len = unsafe { c::lstrlenW(pszPath) };
                // SAFETY: *pszPath is valid up to and including its nul terminator
                Ok(Some(os2path(unsafe { slice::from_raw_parts(pszPath, len as usize) })))
            }
            c::E_FAIL => {
                // This known folder id exists but does not have a path
                if cfg!(debug_assertions) {
                    unreachable!("should not call get_known_folder_path on a virtual folder");
                } else {
                    Err(const_error!(
                        ErrorKind::InvalidInput,
                        "virtual known folders do not have paths"
                    ))
                }
            }
            c::E_INVALIDARG => {
                // This known folder id is not present on the system
                Ok(None)
            }
            _ => {
                // Miscellaneous error
                Err(io::Error::from_raw_os_error(hr))
            }
        };

        // SAFETY: The caller is responsible for freeing the path returned by
        //   SHGetKnownFolderPath by calling CoTaskMemFree, whether it
        //   succeeds or not.
        unsafe { c::CoTaskMemFree(pszPath.cast()) };

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_fetch_known_folder_paths() {
        let dirs = HomeDirs::known_folders().unwrap();
        assert!(dirs.cache_home().is_some());
        assert!(dirs.config_home().is_some());
        assert!(dirs.data_home().is_some());
        assert!(dirs.state_home().is_some());

        let dirs = MediaDirs::known_folders().unwrap();
        assert!(dirs.desktop().is_some());
        assert!(dirs.documents().is_some());
        assert!(dirs.downloads().is_some());
        assert!(dirs.music().is_some());
        assert!(dirs.pictures().is_some());
        assert!(dirs.videos().is_some());
    }
}
