use crate::ffi::{CStr, OsString};
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;

pub struct File(!);

pub struct FileAttr(!);

pub struct ReadDir(!);

pub struct DirEntry(!);

#[allow(unused_variables)]
pub(crate) mod fs_imp {
    use super::unsupported;
    pub(crate) use super::{
        DirBuilder, DirEntry, File, FileAttr, FilePermissions, FileTimes, FileType, OpenOptions,
        ReadDir,
    };
    use crate::io;
    use crate::path::{AsPath, PathBuf};

    pub(crate) fn remove_file<P: AsPath>(path: P) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn symlink_metadata<P: AsPath>(path: P) -> io::Result<FileAttr> {
        unsupported()
    }
    pub(crate) fn metadata<P: AsPath>(path: P) -> io::Result<FileAttr> {
        unsupported()
    }
    pub(crate) fn rename<P: AsPath, Q: AsPath>(from: P, to: Q) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn hard_link<P: AsPath, Q: AsPath>(original: P, link: Q) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn soft_link<P: AsPath, Q: AsPath>(original: P, link: Q) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn remove_dir<P: AsPath>(path: P) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn read_dir<P: AsPath>(path: P) -> io::Result<ReadDir> {
        unsupported()
    }
    pub(crate) fn set_permissions<P: AsPath>(path: P, perms: FilePermissions) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn copy<P: AsPath, Q: AsPath>(from: P, to: Q) -> io::Result<u64> {
        unsupported()
    }
    pub(crate) fn canonicalize<P: AsPath>(path: P) -> io::Result<PathBuf> {
        unsupported()
    }
    pub(crate) fn remove_dir_all<P: AsPath>(path: P) -> io::Result<()> {
        unsupported()
    }
    pub(crate) fn read_link<P: AsPath>(path: P) -> io::Result<PathBuf> {
        unsupported()
    }
    pub(crate) fn try_exists<P: AsPath>(path: P) -> io::Result<bool> {
        unsupported()
    }
}

#[derive(Clone, Debug)]
pub struct OpenOptions {}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {}

pub struct FilePermissions(!);

pub struct FileType(!);

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.0
    }

    pub fn perm(&self) -> FilePermissions {
        self.0
    }

    pub fn file_type(&self) -> FileType {
        self.0
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        self.0
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        self.0
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        self.0
    }
}

impl Clone for FileAttr {
    fn clone(&self) -> FileAttr {
        self.0
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.0
    }

    pub fn set_readonly(&mut self, _readonly: bool) {
        self.0
    }
}

impl Clone for FilePermissions {
    fn clone(&self) -> FilePermissions {
        self.0
    }
}

impl PartialEq for FilePermissions {
    fn eq(&self, _other: &FilePermissions) -> bool {
        self.0
    }
}

impl Eq for FilePermissions {}

impl fmt::Debug for FilePermissions {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, _t: SystemTime) {}
    pub fn set_modified(&mut self, _t: SystemTime) {}
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.0
    }

    pub fn is_file(&self) -> bool {
        self.0
    }

    pub fn is_symlink(&self) -> bool {
        self.0
    }
}

impl Clone for FileType {
    fn clone(&self) -> FileType {
        self.0
    }
}

impl Copy for FileType {}

impl PartialEq for FileType {
    fn eq(&self, _other: &FileType) -> bool {
        self.0
    }
}

impl Eq for FileType {}

impl Hash for FileType {
    fn hash<H: Hasher>(&self, _h: &mut H) {
        self.0
    }
}

impl fmt::Debug for FileType {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        self.0
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.0
    }

    pub fn file_name(&self) -> OsString {
        self.0
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        self.0
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        self.0
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
    pub fn open_native(_path: &CStr, _opts: &OpenOptions) -> io::Result<File> {
        unsupported()
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        self.0
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.0
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.0
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        self.0
    }

    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_read_vectored(&self) -> bool {
        self.0
    }

    pub fn read_buf(&self, _cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.0
    }

    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_write_vectored(&self) -> bool {
        self.0
    }

    pub fn flush(&self) -> io::Result<()> {
        self.0
    }

    pub fn seek(&self, _pos: SeekFrom) -> io::Result<u64> {
        self.0
    }

    pub fn duplicate(&self) -> io::Result<File> {
        self.0
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        self.0
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        self.0
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
        self.0
    }
}
