#![allow(dead_code)] // not used on all platforms

use crate::fs;
use crate::io::{self, Error, ErrorKind};
use crate::path::Path;
use crate::sys_common::ignore_notfound;

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
    if filetype.is_symlink() { fs::remove_file(path) } else { remove_dir_all_recursive(path) }
}

fn remove_dir_all_recursive(path: &Path) -> io::Result<()> {
    for child in fs::read_dir(path)? {
        let result: io::Result<()> = try {
            let child = child?;
            if child.file_type()?.is_dir() {
                remove_dir_all_recursive(&child.path())?;
            } else {
                fs::remove_file(&child.path())?;
            }
        };
        // ignore internal NotFound errors to prevent race conditions
        if let Err(err) = &result
            && err.kind() != io::ErrorKind::NotFound
        {
            return result;
        }
    }
    ignore_notfound(fs::remove_dir(path))
}

pub fn exists(path: &Path) -> io::Result<bool> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}
