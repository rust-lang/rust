use crate::ffi::OsString;
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::io::{self, SeekFrom, IoSlice, IoSliceMut};
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::{unsupported, Void};

pub struct File(Void);

pub struct FileAttr(Void);

pub struct ReadDir(Void);

pub struct DirEntry(Void);

#[derive(Clone, Debug)]
pub struct OpenOptions {}

pub struct FilePermissions(Void);

pub struct FileType(Void);

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        match self.0 {}
    }

    pub fn perm(&self) -> FilePermissions {
        match self.0 {}
    }

    pub fn file_type(&self) -> FileType {
        match self.0 {}
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        match self.0 {}
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        match self.0 {}
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        match self.0 {}
    }
}

impl Clone for FileAttr {
    fn clone(&self) -> FileAttr {
        match self.0 {}
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        match self.0 {}
    }

    pub fn set_readonly(&mut self, _readonly: bool) {
        match self.0 {}
    }
}

impl Clone for FilePermissions {
    fn clone(&self) -> FilePermissions {
        match self.0 {}
    }
}

impl PartialEq for FilePermissions {
    fn eq(&self, _other: &FilePermissions) -> bool {
        match self.0 {}
    }
}

impl Eq for FilePermissions {}

impl fmt::Debug for FilePermissions {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        match self.0 {}
    }

    pub fn is_file(&self) -> bool {
        match self.0 {}
    }

    pub fn is_symlink(&self) -> bool {
        match self.0 {}
    }
}

impl Clone for FileType {
    fn clone(&self) -> FileType {
        match self.0 {}
    }
}

impl Copy for FileType {}

impl PartialEq for FileType {
    fn eq(&self, _other: &FileType) -> bool {
        match self.0 {}
    }
}

impl Eq for FileType {}

impl Hash for FileType {
    fn hash<H: Hasher>(&self, _h: &mut H) {
        match self.0 {}
    }
}

impl fmt::Debug for FileType {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        match self.0 {}
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        match self.0 {}
    }

    pub fn file_name(&self) -> OsString {
        match self.0 {}
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        match self.0 {}
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        match self.0 {}
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {}
    }

    pub fn read(&mut self, _read: bool) {}
    pub fn write(&mut self, _write: bool) {}
    pub fn append(&mut self, _append: bool) {}
    pub fn truncate(&mut self, _truncate: bool) {}
    pub fn create(&mut self, _create: bool) {}
    pub fn create_new(&mut self, _create_new: bool) {}
}

impl File {
    pub fn open(_path: &Path, _opts: &OpenOptions) -> io::Result<File> {
        unsupported()
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        match self.0 {}
    }

    pub fn fsync(&self) -> io::Result<()> {
        match self.0 {}
    }

    pub fn datasync(&self) -> io::Result<()> {
        match self.0 {}
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        match self.0 {}
    }

    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        match self.0 {}
    }

    pub fn flush(&self) -> io::Result<()> {
        match self.0 {}
    }

    pub fn seek(&self, _pos: SeekFrom) -> io::Result<u64> {
        match self.0 {}
    }

    pub fn duplicate(&self) -> io::Result<File> {
        match self.0 {}
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        match self.0 {}
    }

    pub fn diverge(&self) -> ! {
        match self.0 {}
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, _p: &Path) -> io::Result<()> {
        unsupported()
    }
}

impl fmt::Debug for File {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

pub fn readdir(_p: &Path) -> io::Result<ReadDir> {
    unsupported()
}

pub fn unlink(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(_p: &Path, perm: FilePermissions) -> io::Result<()> {
    match perm.0 {}
}

pub fn rmdir(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn remove_dir_all(_path: &Path) -> io::Result<()> {
    unsupported()
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn symlink(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

pub fn stat(_p: &Path) -> io::Result<FileAttr> {
    unsupported()
}

pub fn lstat(_p: &Path) -> io::Result<FileAttr> {
    unsupported()
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn copy(_from: &Path, _to: &Path) -> io::Result<u64> {
    unsupported()
}
