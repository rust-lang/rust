use crate::fs::UserDirs;
use crate::io;
#[cfg(windows)]
use crate::{
    io::{ErrorKind, const_error},
    path::PathBuf,
    ptr, slice,
    sys::{c, os2path},
};

trait Sealed {}
impl Sealed for UserDirs {}

/// Windows-specific extensions to [`fs::UserDirs`](UserDirs).
#[unstable(feature = "dir_discovery", issue = "157515")]
#[expect(private_bounds, reason = "sealed")]
pub trait UserDirsExt: Sized + Sealed {
    /// Load the known user folder paths using the [Known Folders] API.
    ///
    /// The loaded known folders are:
    ///
    /// | `UserDirs` | [`KNOWNFOLDERID`] |
    /// | ---------- | ----------------- |
    /// | [`cache_home`] | [`FOLDERID_LocalAppData`] (`%LOCALAPPDATA%`) |
    /// | [`config_home`] | [`FOLDERID_RoamingAppData`] (`%APPDATA%`) |
    /// | [`data_home`] | [`FOLDERID_RoamingAppData`] (`%APPDATA%`) |
    /// | [`state_home`] | [`FOLDERID_LocalAppData`] (`%LOCALAPPDATA%`) |
    /// | [`desktop`] | [`FOLDERID_Desktop`] (`%USERPROFILE%\Desktop`) |
    /// | [`documents`] | [`FOLDERID_Documents`] (`%USERPROFILE%\Documents`) |
    /// | [`downloads`] | [`FOLDERID_Downloads`] (`%USERPROFILE%\Downloads`) |
    /// | [`music`] | [`FOLDERID_Music`] (`%USERPROFILE%\Music`) |
    /// | [`pictures`] | [`FOLDERID_Pictures`] (`%USERPROFILE%\Pictures`) |
    /// | [`public_share`] | [`FOLDERID_Public`] (`%PUBLIC%`)[^public] |
    /// | [`videos`] | [`FOLDERID_Videos`] (`%USERPROFILE%\Videos`) |
    ///
    /// Note that caches/state are both put in LocalAppData, and config/data
    /// in RoamingAppData. It is always possible for multiple user
    /// directories to be configured to the same path, but this is the common
    /// configuration on Apple platforms, making it even more important to not
    /// assume files in different user directories cannot alias each other.
    ///
    /// [^public]: The default public directory is a separate user folder,
    ///     unlike other OSes where it is a subdirectory of the user's home.
    ///     This is only particularly meaningful on multi-user systems.
    ///
    /// # Errors
    ///
    /// Errors if the underlying system discovery API returns an error.
    /// The lack of a configured path is not considered an error and results
    /// in a `None` value for that path in the returned `UserDirs`.
    ///
    /// # Implementation-specific behavior
    ///
    /// Calls [`SHGetKnownFolderPath`] configured to not create the folder if
    /// missing for the current user once for each of `FOLDERID_Desktop`,
    /// `FOLDERID_Documents`, `FOLDERID_Downloads`, `FOLDERID_LocalAppData`,
    /// `FOLDERID_Music`, `FOLDERID_Pictures`, `FOLDERID_Public`,
    /// `FOLDERID_RoamingAppData`, and `FOLDERID_Videos`.
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
    /// [`FOLDERID_Desktop`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_desktop
    /// [`FOLDERID_Documents`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_documents
    /// [`FOLDERID_Downloads`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_downloads
    /// [`FOLDERID_Music`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_music
    /// [`FOLDERID_Pictures`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_pictures
    /// [`FOLDERID_Public`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_public
    /// [`FOLDERID_Videos`]: https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid#folderid_videos
    #[unstable(feature = "dir_discovery", issue = "157515")]
    fn known_folders() -> io::Result<Self>;
}

#[cfg(windows)]
impl UserDirsExt for UserDirs {
    fn known_folders() -> io::Result<Self> {
        let desktop = get_known_folder_path(&c::FOLDERID_Desktop)?;
        let documents = get_known_folder_path(&c::FOLDERID_Documents)?;
        let downloads = get_known_folder_path(&c::FOLDERID_Downloads)?;
        let local_app_data = get_known_folder_path(&c::FOLDERID_LocalAppData)?;
        let music = get_known_folder_path(&c::FOLDERID_Music)?;
        let pictures = get_known_folder_path(&c::FOLDERID_Pictures)?;
        let public = get_known_folder_path(&c::FOLDERID_Public)?;
        let roaming_app_data = get_known_folder_path(&c::FOLDERID_RoamingAppData)?;
        let videos = get_known_folder_path(&c::FOLDERID_Videos)?;

        // AppData/Local -- system-local, doesn't make sense to sync to another
        // AppData/Roaming -- data that makes sense to sync accross machines

        let mut dirs = UserDirs::empty();

        dirs.home.cache_home = Some(local_app_data.clone());
        dirs.home.config_home = Some(roaming_app_data.clone());
        dirs.home.data_home = Some(roaming_app_data);
        dirs.home.state_home = Some(local_app_data);

        dirs.media.desktop = desktop;
        dirs.media.documents = documents;
        dirs.media.downloads = downloads;
        dirs.media.music = music;
        dirs.media.pictures = pictures;
        dirs.media.public_share = public;
        dirs.media.videos = videos;

        dirs
    }
}

/// Retreive a known folder path from the Windows API.
#[cfg(windows)]
fn get_known_folder_path(id: &c::GUID) -> io::Result<Option<PathBuf>> {
    // Get the known folder path. hToken = NULL requests the current user
    // scope, and we set KF_FLAG_DONT_VERIFY because it's a bit faster and
    // UserDirs does not guarantee that the directories at the paths exist.
    let mut pszPath = ptr::null_mut();
    // SAFETY: rfid/ppszPath are valid pointers, flags are appropriate, and
    //   a NULL hToken is supported by SHGetKnownFolderPath.
    let hr = unsafe {
        c::SHGetKnownFolderPath(
            /* rfid */ id,
            /* dwFlags */ c::KF_FLAG_DONT_VERIFY,
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
            Err(const_error!(ErrorKind::InvalidInput, "virtual known folders do not have paths"))
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

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_fetch_known_folder_paths() {
        let dirs = UserDirs::known_folders().unwrap();
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
