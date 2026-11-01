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
/// - This will not produce an error if the target path is not found.
/// - Like [`std::fs::remove_dir_all`], this helper does not traverse symbolic links, will remove
///   symbolic link itself.
/// - This helper is **not** robust against races on the underlying filesystem, behavior is
///   unspecified if this helper is called concurrently.
/// - This helper is not robust against TOCTOU problems.
///
/// FIXME: Audit whether this implementation is robust enough to replace bootstrap's clean `rm_rf`.
#[track_caller]
pub fn recursive_remove<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let path = path.as_ref();

    // If the path doesn't exist, we treat it as a successful no-op.
    // From the caller's perspective, the goal is simply "ensure this file/dir is gone" —
    // if it's already not there, that's a success, not an error.
    let metadata = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };

    #[cfg(windows)]
    let is_dir_like = |meta: &fs::Metadata| {
        use std::os::windows::fs::FileTypeExt;
        meta.is_dir() || meta.file_type().is_symlink_dir()
    };
    #[cfg(not(windows))]
    let is_dir_like = fs::Metadata::is_dir;

    const MAX_RETRIES: usize = 5;
    const RETRY_DELAY_MS: u64 = 100;

    let try_remove = || {
        if is_dir_like(&metadata) {
            fs::remove_dir_all(path)
        } else {
            try_remove_op_set_perms(fs::remove_file, path, metadata.clone())
        }
    };

    // Retry deletion a few times to handle transient filesystem errors.
    // This is unusual for local file operations, but it's a mitigation
    // against unlikely events where malware scanners may be holding a
    // file beyond our control, to give the malware scanners some opportunity
    // to release their hold.
    for attempt in 0..MAX_RETRIES {
        match try_remove() {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(_) if attempt < MAX_RETRIES - 1 => {
                std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS));
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(())
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

pub fn remove_and_create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let path = path.as_ref();
    recursive_remove(path)?;
    fs::create_dir_all(path)
}
