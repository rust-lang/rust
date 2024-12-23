//! Misc filesystem related helpers for use by bootstrap and tools.
use std::fs::Metadata;
use std::path::Path;
use std::{fs, io};

#[cfg(test)]
mod tests;

/// Helper to ignore [`std::io::ErrorKind::NotFound`], but still propagate other
/// [`std::io::ErrorKind`]s.
pub fn ignore_not_found<Op>(mut op: Op) -> io::Result<()>
where
    Op: FnMut() -> io::Result<()>,
{
    match op() {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// A wrapper around [`std::fs::remove_dir_all`] that can also be used on *non-directory entries*,
/// including files and symbolic links.
///
/// - This will produce an error if the target path is not found.
/// - Like [`std::fs::remove_dir_all`], this helper does not traverse symbolic links, will remove
///   symbolic link itself.
/// - This helper is **not** robust against races on the underlying filesystem, behavior is
///   unspecified if this helper is called concurrently.
/// - This helper is not robust against TOCTOU problems.
///
/// FIXME: this implementation is insufficiently robust to replace bootstrap's clean `rm_rf`
/// implementation:
///
/// - This implementation currently does not perform retries.
#[track_caller]
pub fn recursive_remove<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let path = path.as_ref();
    let metadata = fs::symlink_metadata(path)?;
    #[cfg(windows)]
    let is_dir_like = |meta: &fs::Metadata| {
        use std::os::windows::fs::FileTypeExt;
        meta.is_dir() || meta.file_type().is_symlink_dir()
    };
    #[cfg(not(windows))]
    let is_dir_like = fs::Metadata::is_dir;

    if is_dir_like(&metadata) {
        fs::remove_dir_all(path)
    } else {
        try_remove_op_set_perms(fs::remove_file, path, metadata)
    }
}

fn try_remove_op_set_perms<'p, Op>(mut op: Op, path: &'p Path, metadata: Metadata) -> io::Result<()>
where
    Op: FnMut(&'p Path) -> io::Result<()>,
{
    match op(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
            let mut perms = metadata.permissions();
            perms.set_readonly(false);
            fs::set_permissions(path, perms)?;
            op(path)
        }
        Err(e) => Err(e),
    }
}
