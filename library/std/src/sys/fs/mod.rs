#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unreachable_pub)]

use crate::io;
use crate::path::{Path, PathBuf};

pub(crate) mod common;

cfg_select! {
    any(target_family = "unix", target_os = "wasi") => {
        mod unix;
        use unix as imp;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        pub(super) use unix::CachedFileMetadata;
        #[cfg(not(any(target_os = "fuchsia", target_os = "wasi")))]
        pub(crate) use unix::chroot;
        #[cfg(not(target_os = "wasi"))]
        pub(crate) use unix::debug_assert_fd_is_open;
        #[cfg(not(target_os = "wasi"))]
        pub(crate) use unix::{chown, fchown, lchown, mkfifo};

        use crate::sys::helpers::run_path_with_cstr as with_native_path;
    }
    target_os = "windows" => {
        mod windows;
        use windows as imp;
        pub(crate) use windows::{junction_point, symlink_inner};

        use crate::sys::path::with_native_path;
    }
    target_os = "hermit" => {
        mod hermit;
        use hermit as imp;
    }
    target_os = "motor" => {
        mod motor;
        use motor as imp;
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
    _ => {
        mod unsupported;
        use unsupported as imp;
    }
}

// FIXME: Replace this with platform-specific path conversion functions.
#[cfg(not(any(target_family = "unix", target_os = "windows", target_os = "wasi")))]
#[inline]
pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&Path) -> io::Result<T>) -> io::Result<T> {
    f(path)
}

pub(crate) use imp::{
    Dir, DirBuilder, DirEntry, File, FileAttr, FilePermissions, FileTimes, FileType, OpenOptions,
    ReadDir,
};

pub(crate) fn read_dir(path: &Path) -> io::Result<ReadDir> {
    // FIXME: use with_native_path on all platforms
    imp::readdir(path)
}

pub(crate) fn remove_file(path: &Path) -> io::Result<()> {
    with_native_path(path, &imp::unlink)
}

pub(crate) fn rename(old: &Path, new: &Path) -> io::Result<()> {
    with_native_path(old, &|old| with_native_path(new, &|new| imp::rename(old, new)))
}

pub(crate) fn remove_dir(path: &Path) -> io::Result<()> {
    with_native_path(path, &imp::rmdir)
}

pub(crate) fn remove_dir_all(path: &Path) -> io::Result<()> {
    // FIXME: use with_native_path on all platforms
    #[cfg(not(windows))]
    return imp::remove_dir_all(path);
    #[cfg(windows)]
    with_native_path(path, &imp::remove_dir_all)
}

pub(crate) fn read_link(path: &Path) -> io::Result<PathBuf> {
    with_native_path(path, &imp::readlink)
}

pub(crate) fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    // FIXME: use with_native_path on all platforms
    #[cfg(windows)]
    return imp::symlink(original, link);
    #[cfg(not(windows))]
    with_native_path(original, &|original| {
        with_native_path(link, &|link| imp::symlink(original, link))
    })
}

pub(crate) fn hard_link(original: &Path, link: &Path) -> io::Result<()> {
    with_native_path(original, &|original| {
        with_native_path(link, &|link| imp::link(original, link))
    })
}

pub(crate) fn metadata(path: &Path) -> io::Result<FileAttr> {
    with_native_path(path, &imp::stat)
}

pub(crate) fn symlink_metadata(path: &Path) -> io::Result<FileAttr> {
    with_native_path(path, &imp::lstat)
}

pub(crate) fn set_permissions(path: &Path, perm: FilePermissions) -> io::Result<()> {
    with_native_path(path, &|path| imp::set_perm(path, perm.clone()))
}

pub(crate) fn set_permissions_nofollow(path: &Path, perm: FilePermissions) -> io::Result<()> {
    with_native_path(path, &|path| imp::set_perm_nofollow(path, perm.clone()))
}

pub(crate) fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    with_native_path(path, &imp::canonicalize)
}

pub(crate) fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    // FIXME: use with_native_path on all platforms
    #[cfg(not(windows))]
    return imp::copy(from, to);
    #[cfg(windows)]
    with_native_path(from, &|from| with_native_path(to, &|to| imp::copy(from, to)))
}

pub(crate) fn exists(path: &Path) -> io::Result<bool> {
    // FIXME: use with_native_path on all platforms
    #[cfg(not(windows))]
    return imp::exists(path);
    #[cfg(windows)]
    with_native_path(path, &imp::exists)
}

pub(crate) fn set_times(path: &Path, times: FileTimes) -> io::Result<()> {
    with_native_path(path, &|path| imp::set_times(path, times.clone()))
}

pub(crate) fn set_times_nofollow(path: &Path, times: FileTimes) -> io::Result<()> {
    with_native_path(path, &|path| imp::set_times_nofollow(path, times.clone()))
}
