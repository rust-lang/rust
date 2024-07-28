use std::io;
use std::path::{Path, PathBuf};

// FIXME(jieyouxu): modify create_symlink to panic on windows.

/// Creates a new symlink to a path on the filesystem, adjusting for Windows or Unix.
#[cfg(target_family = "windows")]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    if link.as_ref().exists() {
        std::fs::remove_dir(link.as_ref()).unwrap();
    }
    std::os::windows::fs::symlink_file(original.as_ref(), link.as_ref()).expect(&format!(
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
    std::os::unix::fs::symlink(original.as_ref(), link.as_ref()).expect(&format!(
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
pub fn read_dir_entries<P: AsRef<Path>, F: FnMut(&Path)>(dir: P, mut callback: F) {
    for entry in read_dir(dir) {
        callback(&entry.unwrap().path());
    }
}

/// A wrapper around [`std::fs::remove_file`] which includes the file path in the panic message.
#[track_caller]
pub fn remove_file<P: AsRef<Path>>(path: P) {
    std::fs::remove_file(path.as_ref())
        .expect(&format!("the file in path \"{}\" could not be removed", path.as_ref().display()));
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

/// A wrapper around [`std::fs::metadata`] which includes the file path in the panic message.
#[track_caller]
pub fn metadata<P: AsRef<Path>>(path: P) -> std::fs::Metadata {
    std::fs::metadata(path.as_ref()).expect(&format!(
        "the file's metadata in path \"{}\" could not be read",
        path.as_ref().display()
    ))
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

/// A function which prints all file names in the directory `dir` similarly to Unix's `ls`.
/// Useful for debugging.
/// Usage: `eprintln!("{:#?}", shallow_find_dir_entries(some_dir));`
#[track_caller]
pub fn shallow_find_dir_entries<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    let paths = read_dir(dir);
    let mut output = Vec::new();
    for path in paths {
        output.push(path.unwrap().path());
    }
    output
}
