use std::io;
use std::path::Path;

use crate::fs_wrapper;

// FIXME(jieyouxu): modify create_symlink to panic on windows.

/// Creates a new symlink to a path on the filesystem, adjusting for Windows or Unix.
#[cfg(target_family = "windows")]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    if link.as_ref().exists() {
        std::fs::remove_dir(link.as_ref()).unwrap();
    }
    use std::os::windows::fs;
    fs::symlink_file(original.as_ref(), link.as_ref()).expect(&format!(
        "failed to create symlink {:?} for {:?}",
        link.as_ref().display(),
        original.as_ref().display(),
    ));
}

/// Creates a new symlink to a path on the filesystem, adjusting for Windows or Unix.
#[cfg(target_family = "unix")]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    if link.as_ref().exists() {
        std::fs::remove_dir(link.as_ref()).unwrap();
    }
    use std::os::unix::fs;
    fs::symlink(original.as_ref(), link.as_ref()).expect(&format!(
        "failed to create symlink {:?} for {:?}",
        link.as_ref().display(),
        original.as_ref().display(),
    ));
}

/// Copy a directory into another.
pub fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    fn copy_dir_all_inner(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
        let dst = dst.as_ref();
        if !dst.is_dir() {
            std::fs::create_dir_all(&dst)?;
        }
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            if ty.is_dir() {
                copy_dir_all_inner(entry.path(), dst.join(entry.file_name()))?;
            } else {
                std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
            }
        }
        Ok(())
    }

    if let Err(e) = copy_dir_all_inner(&src, &dst) {
        // Trying to give more context about what exactly caused the failure
        panic!(
            "failed to copy `{}` to `{}`: {:?}",
            src.as_ref().display(),
            dst.as_ref().display(),
            e
        );
    }
}

/// Helper for reading entries in a given directory.
pub fn read_dir<P: AsRef<Path>, F: FnMut(&Path)>(dir: P, mut callback: F) {
    for entry in fs_wrapper::read_dir(dir) {
        callback(&entry.unwrap().path());
    }
}
