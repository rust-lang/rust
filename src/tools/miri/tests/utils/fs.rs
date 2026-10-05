use std::ffi::{CStr, OsString};
use std::path::PathBuf;
use std::{env, fs, io};

use super::{into_c_string, miri_extern};

pub fn host_to_target_path(path: OsString) -> PathBuf {
    let path = into_c_string(path);
    let mut out = Vec::with_capacity(1024);

    unsafe {
        let ret =
            miri_extern::miri_host_to_target_path(path.as_ptr(), out.as_mut_ptr(), out.capacity());
        assert_eq!(ret, 0);
        // Here we panic if it's not UTF-8... but that is hard to avoid with OsStr APIs.
        let out = CStr::from_ptr(out.as_ptr()).to_str().unwrap();
        PathBuf::from(out)
    }
}

/// You probably want to use `prepare` instead to ensure that the file is fresh!
pub fn tmp() -> PathBuf {
    let path =
        std::env::var_os("MIRI_TEMP").unwrap_or_else(|| std::env::temp_dir().into_os_string());
    // These are host paths. We need to convert them to the target.
    host_to_target_path(path)
}

/// Prepare: compute filename and make sure the file does not exist.
pub fn prepare(filename: &str) -> PathBuf {
    assert!(filename.starts_with("miri"));

    let path = tmp().join(filename);
    // Clean the paths for robustness.
    fs::remove_file(&path).ok();
    path
}

/// Prepare like above, and also write some initial content to the file.
pub fn prepare_with_content(filename: &str, content: &[u8]) -> PathBuf {
    let path = prepare(filename);
    fs::write(&path, content).unwrap();
    path
}

/// Prepare directory: compute directory name and make sure it does not exist.
pub fn prepare_dir(dirname: &str) -> PathBuf {
    assert!(dirname.starts_with("miri"));

    let path = tmp().join(&dirname);
    // Clean the directory for robustness.
    fs::remove_dir_all(&path).ok();
    path
}

/// Windows makes things difficult by refusing create symlinks per default. GHA is configured to
/// allow them, but when people run the tests on their systems we'd prefer them to pass without
/// special setup. So we try to detect whether symlinks are working. To make things extra fun, this
/// can happen even if we *think* we are on Unix since the host could still be Windows.
pub fn have_symlink_permission() -> bool {
    use std::sync::LazyLock;

    static HAVE_SYMLINK_PERMISSION: LazyLock<bool> = LazyLock::new(|| {
        // Never skip any tests on CI.
        if env::var_os("CI").is_some() {
            return true;
        }

        #[cfg(unix)]
        use std::os::unix::fs::symlink as symlink_file;
        #[cfg(windows)]
        use std::os::windows::fs::symlink_file;

        let link = prepare("miri_have_symlink_permission_check_file");
        if symlink_file(r"nonexisting_target", &link).is_ok() {
            // Looking pretty good.
            fs::remove_file(link).unwrap();
            return true;
        }
        // Looking bad. But just to confirm, could we create a normal file?
        if fs::write(&link, &[]).is_ok() {
            // Normal file works, symlink did not -- looks like the Windows issue.
            fs::remove_file(link).unwrap();
            return false;
        }
        panic!("unable to create files in tempdir");
    });
    *HAVE_SYMLINK_PERMISSION
}
