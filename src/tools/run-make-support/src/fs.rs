use std::fs::FileType;
use std::io;
use std::path::{Path, PathBuf};

/// Given a symlink at `src`, read its target, then create a new symlink at `dst` also pointing to
/// target.
pub fn copy_symlink(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    let src = src.as_ref();
    let dst = dst.as_ref();
    let metadata = symlink_metadata(src);
    if let Err(e) = copy_symlink_raw(metadata.file_type(), src, dst) {
        panic!("failed to copy symlink from `{}` to `{}`: {e}", src.display(), dst.display(),);
    }
}

fn copy_symlink_raw(ty: FileType, src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
    // Traverse symlink once to find path of target entity.
    let target_path = std::fs::read_link(src)?;

    let new_symlink_path = dst.as_ref();
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileTypeExt;
        if ty.is_symlink_dir() {
            std::os::windows::fs::symlink_dir(&target_path, new_symlink_path)?;
        } else {
            // Target may be a file or another symlink, in any case we can use
            // `symlink_file` here.
            std::os::windows::fs::symlink_file(&target_path, new_symlink_path)?;
        }
    }
    #[cfg(unix)]
    {
        let _ = ty;
        std::os::unix::fs::symlink(target_path, new_symlink_path)?;
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = ty;
        // Technically there's also wasi, but I have no clue about wasi symlink
        // semantics and which wasi targets / environment support symlinks.
        unimplemented!("unsupported target");
    }
    Ok(())
}

/// Copy a directory into another. This will not traverse symlinks; instead, it will create new
/// symlinks pointing at target paths that symlinks in the original directory points to.
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
            } else if ty.is_symlink() {
                copy_symlink_raw(ty, entry.path(), dst.join(entry.file_name()))?;
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
pub fn read_dir_entries<P: AsRef<Path>, F: FnMut(&Path)>(dir: P, mut callback: F) {
    for entry in read_dir(dir) {
        callback(&entry.unwrap().path());
    }
}

/// A wrapper around [`build_helper::fs::recursive_remove`] which includes the file path in the
/// panic message.
///
/// This handles removing symlinks on Windows (e.g. symlink-to-file will be removed via
/// [`std::fs::remove_file`] while symlink-to-dir will be removed via [`std::fs::remove_dir`]).
#[track_caller]
pub fn recursive_remove<P: AsRef<Path>>(path: P) {
    if let Err(e) = build_helper::fs::recursive_remove(path.as_ref()) {
        panic!(
            "failed to recursive remove filesystem entities at `{}`: {e}",
            path.as_ref().display()
        );
    }
}

/// A wrapper around [`std::fs::remove_file`] which includes the file path in the panic message.
#[track_caller]
pub fn remove_file<P: AsRef<Path>>(path: P) {
    if let Err(e) = std::fs::remove_file(path.as_ref()) {
        panic!("failed to remove file at `{}`: {e}", path.as_ref().display());
    }
}

/// A wrapper around [`std::fs::remove_dir`] which includes the directory path in the panic message.
#[track_caller]
pub fn remove_dir<P: AsRef<Path>>(path: P) {
    if let Err(e) = std::fs::remove_dir(path.as_ref()) {
        panic!("failed to remove directory at `{}`: {e}", path.as_ref().display());
    }
}

/// A wrapper around [`std::fs::copy`] which includes the file path in the panic message.
#[track_caller]
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) {
    std::fs::copy(from.as_ref(), to.as_ref()).expect(&format!(
        "the file \"{}\" could not be copied over to \"{}\"",
        from.as_ref().display(),
        to.as_ref().display(),
    ));
}

/// A wrapper around [`std::fs::File::create`] which includes the file path in the panic message.
#[track_caller]
pub fn create_file<P: AsRef<Path>>(path: P) {
    std::fs::File::create(path.as_ref())
        .expect(&format!("the file in path \"{}\" could not be created", path.as_ref().display()));
}

/// A wrapper around [`std::fs::read`] which includes the file path in the panic message.
#[track_caller]
pub fn read<P: AsRef<Path>>(path: P) -> Vec<u8> {
    std::fs::read(path.as_ref())
        .expect(&format!("the file in path \"{}\" could not be read", path.as_ref().display()))
}

/// A wrapper around [`std::fs::read_to_string`] which includes the file path in the panic message.
#[track_caller]
pub fn read_to_string<P: AsRef<Path>>(path: P) -> String {
    std::fs::read_to_string(path.as_ref()).expect(&format!(
        "the file in path \"{}\" could not be read into a String",
        path.as_ref().display()
    ))
}

/// A wrapper around [`std::fs::read_dir`] which includes the file path in the panic message.
#[track_caller]
pub fn read_dir<P: AsRef<Path>>(path: P) -> std::fs::ReadDir {
    std::fs::read_dir(path.as_ref())
        .expect(&format!("the directory in path \"{}\" could not be read", path.as_ref().display()))
}

/// A wrapper around [`std::fs::write`] which includes the file path in the panic message.
#[track_caller]
pub fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) {
    std::fs::write(path.as_ref(), contents.as_ref()).expect(&format!(
        "the file in path \"{}\" could not be written to",
        path.as_ref().display()
    ));
}

/// A wrapper around [`std::fs::remove_dir_all`] which includes the file path in the panic message.
#[track_caller]
pub fn remove_dir_all<P: AsRef<Path>>(path: P) {
    std::fs::remove_dir_all(path.as_ref()).expect(&format!(
        "the directory in path \"{}\" could not be removed alongside all its contents",
        path.as_ref().display(),
    ));
}

/// A wrapper around [`std::fs::create_dir`] which includes the file path in the panic message.
#[track_caller]
pub fn create_dir<P: AsRef<Path>>(path: P) {
    std::fs::create_dir(path.as_ref()).expect(&format!(
        "the directory in path \"{}\" could not be created",
        path.as_ref().display()
    ));
}

/// A wrapper around [`std::fs::create_dir_all`] which includes the file path in the panic message.
#[track_caller]
pub fn create_dir_all<P: AsRef<Path>>(path: P) {
    std::fs::create_dir_all(path.as_ref()).expect(&format!(
        "the directory (and all its parents) in path \"{}\" could not be created",
        path.as_ref().display()
    ));
}

/// A wrapper around [`std::fs::metadata`] which includes the file path in the panic message. Note
/// that this will traverse symlinks and will return metadata about the target file. Use
/// [`symlink_metadata`] if you don't want to traverse symlinks.
///
/// See [`std::fs::metadata`] docs for more details.
#[track_caller]
pub fn metadata<P: AsRef<Path>>(path: P) -> std::fs::Metadata {
    match std::fs::metadata(path.as_ref()) {
        Ok(m) => m,
        Err(e) => panic!("failed to read file metadata at `{}`: {e}", path.as_ref().display()),
    }
}

/// A wrapper around [`std::fs::symlink_metadata`] which includes the file path in the panic
/// message. Note that this will not traverse symlinks and will return metadata about the filesystem
/// entity itself. Use [`metadata`] if you want to traverse symlinks.
///
/// See [`std::fs::symlink_metadata`] docs for more details.
#[track_caller]
pub fn symlink_metadata<P: AsRef<Path>>(path: P) -> std::fs::Metadata {
    match std::fs::symlink_metadata(path.as_ref()) {
        Ok(m) => m,
        Err(e) => {
            panic!("failed to read file metadata (shallow) at `{}`: {e}", path.as_ref().display())
        }
    }
}

/// A wrapper around [`std::fs::rename`] which includes the file path in the panic message.
#[track_caller]
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) {
    std::fs::rename(from.as_ref(), to.as_ref()).expect(&format!(
        "the file \"{}\" could not be moved over to \"{}\"",
        from.as_ref().display(),
        to.as_ref().display(),
    ));
}

/// A wrapper around [`std::fs::set_permissions`] which includes the file path in the panic message.
#[track_caller]
pub fn set_permissions<P: AsRef<Path>>(path: P, perm: std::fs::Permissions) {
    std::fs::set_permissions(path.as_ref(), perm).expect(&format!(
        "the file's permissions in path \"{}\" could not be changed",
        path.as_ref().display()
    ));
}

/// List directory entries immediately under the given `dir`.
#[track_caller]
pub fn shallow_find_dir_entries<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    let paths = read_dir(dir);
    let mut output = Vec::new();
    for path in paths {
        output.push(path.unwrap().path());
    }
    output
}

/// Create a new symbolic link to a directory.
///
/// # Removing the symlink
///
/// - On Windows, a symlink-to-directory needs to be removed with a corresponding [`fs::remove_dir`]
///   and not [`fs::remove_file`].
/// - On Unix, remove the symlink with [`fs::remove_file`].
///
/// [`fs::remove_dir`]: crate::fs::remove_dir
/// [`fs::remove_file`]: crate::fs::remove_file
pub fn symlink_dir<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    #[cfg(unix)]
    {
        if let Err(e) = std::os::unix::fs::symlink(original.as_ref(), link.as_ref()) {
            panic!(
                "failed to create symlink: original=`{}`, link=`{}`: {e}",
                original.as_ref().display(),
                link.as_ref().display()
            );
        }
    }
    #[cfg(windows)]
    {
        if let Err(e) = std::os::windows::fs::symlink_dir(original.as_ref(), link.as_ref()) {
            panic!(
                "failed to create symlink-to-directory: original=`{}`, link=`{}`: {e}",
                original.as_ref().display(),
                link.as_ref().display()
            );
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        unimplemented!("target family not currently supported")
    }
}

/// Create a new symbolic link to a file.
///
/// # Removing the symlink
///
/// On both Windows and Unix, a symlink-to-file needs to be removed with a corresponding
/// [`fs::remove_file`](crate::fs::remove_file) and not [`fs::remove_dir`](crate::fs::remove_dir).
pub fn symlink_file<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    #[cfg(unix)]
    {
        if let Err(e) = std::os::unix::fs::symlink(original.as_ref(), link.as_ref()) {
            panic!(
                "failed to create symlink: original=`{}`, link=`{}`: {e}",
                original.as_ref().display(),
                link.as_ref().display()
            );
        }
    }
    #[cfg(windows)]
    {
        if let Err(e) = std::os::windows::fs::symlink_file(original.as_ref(), link.as_ref()) {
            panic!(
                "failed to create symlink-to-file: original=`{}`, link=`{}`: {e}",
                original.as_ref().display(),
                link.as_ref().display()
            );
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        unimplemented!("target family not currently supported")
    }
}
