use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::errors::AppleSdkError;

#[cfg(test)]
mod tests;

// TOCTOU is not _really_ an issue with our use of `try_exists` in here, we mostly use it for
// diagnostics, and these directories are global state that the user may change anytime anyhow.
fn try_exists(path: &Path) -> Result<bool, AppleSdkError> {
    path.try_exists().map_err(|error| AppleSdkError::FailedReading { path: path.to_owned(), error })
}

/// Get the SDK path for an SDK under `/Library/Developer/CommandLineTools`.
fn sdk_root_in_sdks_dir(sdks_dir: impl Into<PathBuf>, sdk_name: &str) -> PathBuf {
    let mut path = sdks_dir.into();
    path.push("SDKs");
    path.push(sdk_name);
    path.set_extension("sdk");
    path
}

/// Get the SDK path for an SDK under `/Applications/Xcode.app/Contents/Developer`.
fn sdk_root_in_developer_dir(developer_dir: impl Into<PathBuf>, sdk_name: &str) -> PathBuf {
    let mut path = developer_dir.into();
    path.push("Platforms");
    path.push(sdk_name);
    path.set_extension("platform");
    path.push("Developer");
    path.push("SDKs");
    path.push(sdk_name);
    path.set_extension("sdk");
    path
}

/// Find a SDK root from the user's environment for the given SDK name.
///
/// We do this by searching (purely by names in the filesystem, without reading SDKSettings.json)
/// for a matching SDK in the following places:
/// - `DEVELOPER_DIR`
/// - `/var/db/xcode_select_link`
/// - `/Applications/Xcode.app`
/// - `/Library/Developer/CommandLineTools`
///
/// This does roughly the same thing as `xcrun -sdk $sdk_name -show-sdk-path` (see `man xcrun` for
/// a few details on the search algorithm).
///
/// The reason why we implement this logic ourselves is:
/// - Reading these directly is faster than spawning a new process.
/// - `xcrun` can be fairly slow to start up after a reboot.
/// - In the future, we will be able to integrate this better with the compiler's change tracking
///   mechanisms, allowing rebuilds when the involved env vars and paths here change. See #118204.
/// - It's easier for us to emit better error messages.
///
/// Though a downside is that `xcrun` might be expanded in the future to check more places, and then
/// `rustc` would have to be changed to keep up. Furthermore, `xcrun`'s exact algorithm is
/// undocumented, so it might be doing more things than we do here.
pub(crate) fn find_sdk_root(sdk_name: &'static str) -> Result<PathBuf, AppleSdkError> {
    // Only try this if host OS is macOS.
    if !cfg!(target_os = "macos") {
        return Err(AppleSdkError::MissingCrossCompileNonMacOS { sdk_name });
    }

    // NOTE: We could consider walking upwards in `SDKROOT` assuming Xcode directory structure, but
    // that isn't what `xcrun` does, and might still not yield the desired result (e.g. if using an
    // old SDK to compile for an old ARM iOS arch, we don't want `rustc` to pick a macOS SDK from
    // the old Xcode).

    // Try reading from `DEVELOPER_DIR` on all hosts.
    if let Some(dir) = std::env::var_os("DEVELOPER_DIR") {
        let dir = PathBuf::from(dir);
        let sdkroot = sdk_root_in_developer_dir(&dir, sdk_name);

        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            let sdkroot_bare = sdk_root_in_sdks_dir(&dir, sdk_name);
            if try_exists(&sdkroot_bare)? {
                return Ok(sdkroot_bare);
            } else {
                return Err(AppleSdkError::MissingDeveloperDir { dir, sdkroot, sdkroot_bare });
            }
        }
    }

    // Next, try to read the link that `xcode-select` sets.
    //
    // FIXME(madsmtm): Support cases where `/var/db/xcode_select_link` contains a relative path?
    let path = PathBuf::from("/var/db/xcode_select_link");
    match fs::read_link(&path) {
        Ok(dir) => {
            let sdkroot = sdk_root_in_developer_dir(&dir, sdk_name);
            if try_exists(&sdkroot)? {
                return Ok(sdkroot);
            } else {
                let sdkroot_bare = sdk_root_in_sdks_dir(&dir, sdk_name);
                if try_exists(&sdkroot_bare)? {
                    return Ok(sdkroot_bare);
                } else {
                    return Err(AppleSdkError::MissingXcodeSelect { dir, sdkroot, sdkroot_bare });
                }
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            // Intentionally ignore not found errors, if `xcode-select --reset` is called the
            // link will not exist.
        }
        Err(error) => return Err(AppleSdkError::FailedReading { path, error }),
    }

    // Next, fall back to reading from `/Applications/Xcode.app`.
    let dir = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
    if try_exists(&dir)? {
        let sdkroot = sdk_root_in_developer_dir(&dir, sdk_name);
        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            return Err(AppleSdkError::MissingXcode { sdkroot, sdk_name });
        }
    }

    // Finally, fall back to reading from `/Library/Developer/CommandLineTools`.
    let dir = PathBuf::from("/Library/Developer/CommandLineTools");
    if try_exists(&dir)? {
        let sdkroot = sdk_root_in_sdks_dir(&dir, sdk_name);
        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            return Err(AppleSdkError::MissingCommandlineTools { sdkroot, sdk_name });
        }
    }

    Err(AppleSdkError::Missing { sdk_name })
}
