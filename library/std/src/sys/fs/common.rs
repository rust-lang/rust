#![allow(dead_code)] // not used on all platforms
use alloc_crate::collections::VecDeque;

use crate::io::{self, Error, ErrorKind};
use crate::path::{Path, PathBuf};
use crate::sys::fs::{File, OpenOptions};
use crate::sys::helpers::ignore_notfound;
use crate::{fmt, fs};

pub(crate) const NOT_FILE_ERROR: Error = io::const_error!(
    ErrorKind::InvalidInput,
    "the source path is neither a regular file nor a symlink to a regular file",
);

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    let mut reader = fs::File::open(from)?;
    let metadata = reader.metadata()?;

    if !metadata.is_file() {
        return Err(NOT_FILE_ERROR);
    }

    let mut writer = fs::File::create(to)?;
    let perm = metadata.permissions();

    let ret = io::copy(&mut reader, &mut writer)?;
    writer.set_permissions(perm)?;
    Ok(ret)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let filetype = fs::symlink_metadata(path)?.file_type();
    if filetype.is_symlink() { fs::remove_file(path) } else { remove_dir_all_iterative(path) }
}

fn remove_dir_all_iterative(path: &Path) -> io::Result<()> {
    // In unix/windows/solid/hermit/motor/uefi implementation of ReadDir struct, there is a field
    // called `root` that contains the directory path that we are reading directory entries
    // from. This makes holding a PathBuf in this tuple redundant since we only need this to remove
    // the directory when we have exhausted the ReadDir iterator; if we can expose that field
    // in the ReadDir struct through a method that return &Path, we can reduce memory usage
    // allocated to this VecDeque.

    let mut directories = VecDeque::new();
    directories.push_front((path.to_path_buf(), fs::read_dir(path)?));

    while !directories.is_empty() {
        let (parent_path, read_dir) = &mut directories[0];
        let child = read_dir.next();
        if let Some(child) = child {
            let result: io::Result<()> = try {
                let child = child?;
                let child_path = child.path();
                if child.file_type()?.is_dir() {
                    let child_readdir = fs::read_dir(&child_path)?;
                    directories.push_front((child_path, child_readdir));
                } else {
                    fs::remove_file(&child_path)?;
                }
            };

            // ignore internal NotFound errors to prevent race conditions
            if let Err(err) = &result
                && err.kind() != io::ErrorKind::NotFound
            {
                return result;
            }
        } else {
            ignore_notfound(fs::remove_dir(parent_path))?;
            directories.pop_front();
        }
    }

    Ok(())
}

pub fn exists(path: &Path) -> io::Result<bool> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

pub struct Dir {
    path: PathBuf,
}

impl Dir {
    pub fn open(path: &Path, _opts: &OpenOptions) -> io::Result<Self> {
        path.canonicalize().map(|path| Self { path })
    }

    pub fn open_file(&self, path: &Path, opts: &OpenOptions) -> io::Result<File> {
        File::open(&self.path.join(path), &opts)
    }
}

impl fmt::Debug for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dir").field("path", &self.path).finish()
    }
}
