#![allow(dead_code)] // not used on all platforms

use crate::path::Path;
use crate::fs;
use crate::io::{self, Error, ErrorKind};

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    if !from.is_file() {
        return Err(Error::new(ErrorKind::InvalidInput,
                              "the source path is not an existing regular file"))
    }

    let mut reader = fs::File::open(from)?;
    let mut writer = fs::File::create(to)?;
    let perm = reader.metadata()?.permissions();

    let ret = io::copy(&mut reader, &mut writer)?;
    fs::set_permissions(to, perm)?;
    Ok(ret)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let filetype = fs::symlink_metadata(path)?.file_type();
    if filetype.is_symlink() {
        fs::remove_file(path)
    } else {
        remove_dir_all_recursive(path)
    }
}

fn remove_dir_all_recursive(path: &Path) -> io::Result<()> {
    for child in fs::read_dir(path)? {
        let child = child?;
        if child.file_type()?.is_dir() {
            remove_dir_all_recursive(&child.path())?;
        } else {
            fs::remove_file(&child.path())?;
        }
    }
    fs::remove_dir(path)
}
