#![deny(unsafe_op_in_unsafe_fn)]

use crate::io;
use crate::path::{Path, PathBuf};

pub mod common;

cfg_select! {
    target_family = "unix" => {
        mod unix;
        use unix as imp;
        pub use unix::{chown, fchown, lchown, mkfifo};
        #[cfg(not(target_os = "fuchsia"))]
        pub use unix::chroot;
        pub(crate) use unix::debug_assert_fd_is_open;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        pub(crate) use unix::CachedFileMetadata;
        use crate::sys::common::small_c_string::run_path_with_cstr as with_native_path;
    }
    target_os = "windows" => {
        mod windows;
        use windows as imp;
        pub use windows::{symlink_inner, junction_point};
        use crate::sys::path::with_native_path;
    }
    target_os = "hermit" => {
        mod hermit;
        use hermit as imp;
    }
    target_os = "solid_asp3" => {
        mod solid;
        use solid as imp;
    }
    target_os = "uefi" => {
        mod uefi;
        use uefi as imp;
    }
    target_os = "vexos" => {
        mod vexos;
        use vexos as imp;
    }
    target_os = "wasi" => {
        mod wasi;
        use wasi as imp;
    }
    _ => {
        mod unsupported;
        use unsupported as imp;
    }
}

// FIXME: Replace this with platform-specific path conversion functions.
#[cfg(not(any(target_family = "unix", target_os = "windows")))]
#[inline]
pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&Path) -> io::Result<T>) -> io::Result<T> {
    f(path)
}

pub use imp::{
    DirBuilder, DirEntry, File, FileAttr, FilePermissions, FileTimes, FileType, OpenOptions,
    ReadDir,
};

pub fn read_dir(path: &Path) -> io::Result<ReadDir> {
    // FIXME: use with_native_path on all platforms
    imp::readdir(path)
}

pub fn remove_file(path: &Path) -> io::Result<()> {
    with_native_path(path, &imp::unlink)
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    with_native_path(old, &|old| with_native_path(new, &|new| imp::rename(old, new)))
}

pub fn remove_dir(path: &Path) -> io::Result<()> {
    with_native_path(path, &imp::rmdir)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    // FIXME: use with_native_path on all platforms
    #[cfg(not(windows))]
    return imp::remove_dir_all(path);
    #[cfg(windows)]
    with_native_path(path, &imp::remove_dir_all)
}

pub fn read_link(path: &Path) -> io::Result<PathBuf> {
    with_native_path(path, &imp::readlink)
}

pub fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    // FIXME: use with_native_path on all platforms
    #[cfg(windows)]
    return imp::symlink(original, link);
    #[cfg(not(windows))]
    with_native_path(original, &|original| {
        with_native_path(link, &|link| imp::symlink(original, link))
    })
}

pub fn hard_link(original: &Path, link: &Path) -> io::Result<()> {
    with_native_path(original, &|original| {
        with_native_path(link, &|link| imp::link(original, link))
    })
}

pub fn metadata(path: &Path) -> io::Result<FileAttr> {
    with_native_path(path, &imp::stat)
}

pub fn symlink_metadata(path: &Path) -> io::Result<FileAttr> {
    with_native_path(path, &imp::lstat)
}

pub fn set_permissions(path: &Path, perm: FilePermissions) -> io::Result<()> {
    with_native_path(path, &|path| imp::set_perm(path, perm.clone()))
}

#[cfg(unix)]
pub fn set_permissions_nofollow(path: &Path, perm: crate::fs::Permissions) -> io::Result<()> {
    use crate::fs::OpenOptions;

    let mut options = OpenOptions::new();

    // ESP-IDF and Horizon do not support O_NOFOLLOW, so we skip setting it.
    // Their filesystems do not have symbolic links, so no special handling is required.
    #[cfg(not(any(target_os = "espidf", target_os = "horizon")))]
    {
        use crate::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }

    options.open(path)?.set_permissions(perm)
}

#[cfg(not(unix))]
pub fn set_permissions_nofollow(_path: &Path, _perm: crate::fs::Permissions) -> io::Result<()> {
    crate::unimplemented!(
        "`set_permissions_nofollow` is currently only implemented on Unix platforms"
    )
}

pub fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    with_native_path(path, &imp::canonicalize)
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    // FIXME: use with_native_path on all platforms
    #[cfg(not(windows))]
    return imp::copy(from, to);
    #[cfg(windows)]
    with_native_path(from, &|from| with_native_path(to, &|to| imp::copy(from, to)))
}

pub fn exists(path: &Path) -> io::Result<bool> {
    // FIXME: use with_native_path on all platforms
    #[cfg(not(windows))]
    return imp::exists(path);
    #[cfg(windows)]
    with_native_path(path, &imp::exists)
}
