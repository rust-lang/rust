#![allow(dead_code)] // not used on all platforms

use crate::fmt;
use crate::fs::{self, create_dir, remove_dir, remove_file, rename};
use crate::io::{self, Error, ErrorKind};
use crate::path::{Path, PathBuf};
use crate::sys::fs::{File, OpenOptions, symlink};
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

pub struct Dir {
    path: PathBuf,
}

impl Dir {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self { path: path.as_ref().to_path_buf() })
    }

    pub fn new_with<P: AsRef<Path>>(path: P, _opts: &OpenOptions) -> io::Result<Self> {
        Ok(Self { path: path.as_ref().to_path_buf() })
    }

    pub fn new_for_traversal<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self { path: path.as_ref().to_path_buf() })
    }

    pub fn open<P: AsRef<Path>>(&self, path: P) -> io::Result<File> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        File::open(&self.path.join(path), &opts)
    }

    pub fn open_with<P: AsRef<Path>>(&self, path: P, opts: &OpenOptions) -> io::Result<File> {
        File::open(&self.path.join(path), opts)
    }

    pub fn create_dir<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        create_dir(self.path.join(path))
    }

    pub fn open_dir<P: AsRef<Path>>(&self, path: P) -> io::Result<Self> {
        Self::new(self.path.join(path))
    }

    pub fn open_dir_with<P: AsRef<Path>>(&self, path: P, opts: &OpenOptions) -> io::Result<Self> {
        Self::new_with(self.path.join(path), opts)
    }

    pub fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        remove_file(self.path.join(path))
    }

    pub fn remove_dir<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        remove_dir(self.path.join(path))
    }

    pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        from: P,
        to_dir: &Self,
        to: Q,
    ) -> io::Result<()> {
        rename(self.path.join(from), to_dir.path.join(to))
    }

    pub fn symlink<P: AsRef<Path>, Q: AsRef<Path>>(&self, original: P, link: Q) -> io::Result<()> {
        symlink(original.as_ref(), link.as_ref())
    }
}

impl fmt::Debug for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dir").field("path", &self.path).finish()
    }
}
