#![allow(missing_docs)]

use crate::custom_os_impl;
use crate::ffi::OsString;
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};

use crate::os::custom::time::SystemTime;

/// Inner content of [`crate::fs::Metadata`]
#[derive(Debug, Clone)]
pub struct FileAttr {
    size: u64,
    perm: FilePermissions,
    file_type: FileType,
    modified: Option<SystemTime>,
    accessed: Option<SystemTime>,
    created: Option<SystemTime>,
}

fn time_error() -> io::Error {
    let msg = "Couldn't obtain file modification/access/creation time";
    io::Error::new(io::ErrorKind::Unsupported, msg)
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn perm(&self) -> FilePermissions {
        self.perm
    }

    pub fn file_type(&self) -> FileType {
        self.file_type
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        self.modified.ok_or_else(time_error)
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        self.accessed.ok_or_else(time_error)
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        self.created.ok_or_else(time_error)
    }
}

/// Inner content of [`crate::fs::Permissions`]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FilePermissions {
    pub read_only: bool,
}

impl FilePermissions {
    pub(crate) fn readonly(&self) -> bool {
        self.read_only
    }

    pub(crate) fn set_readonly(&mut self, readonly: bool) {
        self.read_only = readonly;
    }
}

/// Inner content of [`crate::fs::FileTimes`]
#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    pub accessed: Option<SystemTime>,
    pub modified: Option<SystemTime>,
}

impl FileTimes {
    pub(crate) fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = Some(t);
    }

    pub(crate) fn set_modified(&mut self, t: SystemTime) {
        self.modified = Some(t);
    }
}

/// Inner content of [`crate::fs::FileType`]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileType {
    pub symlink: bool,
    pub directory: bool,
}

impl FileType {
    pub(crate) fn is_dir(&self) -> bool {
        self.directory
    }

    pub(crate) fn is_file(&self) -> bool {
        !self.directory
    }

    pub(crate) fn is_symlink(&self) -> bool {
        self.symlink
    }
}

/// Inner content of [`crate::fs::ReadDir`]
pub type ReadDir = Box<dyn ReadDirApi>;

/// Wrapper around `Iterator<DirEntry>` and `Debug`
pub trait ReadDirApi: Iterator<Item = io::Result<DirEntry>> + core::fmt::Debug {}

/// Inner content of [`crate::fs::DirEntry`]
pub struct DirEntry {
    pub path: PathBuf,
    pub file_name: OsString,
    pub metadata: FileAttr,
    pub file_type: FileType,
}

impl DirEntry {
    pub(crate) fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub(crate) fn file_name(&self) -> OsString {
        self.file_name.clone()
    }

    pub(crate) fn metadata(&self) -> io::Result<FileAttr> {
        Ok(self.metadata.clone())
    }

    pub(crate) fn file_type(&self) -> io::Result<FileType> {
        Ok(self.file_type.clone())
    }
}

/// Inner content of [`crate::fs::OpenOptions`]
#[derive(Copy, Clone, Debug, Default)]
pub struct OpenOptions {
    pub read: bool,
    pub write: bool,
    pub append: bool,
    pub truncate: bool,
    pub create: bool,
    pub create_new: bool,
}

impl OpenOptions {
    pub(crate) fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
        }
    }

    pub(crate) fn read(&mut self, read: bool) {
        self.read = read;
    }

    pub(crate) fn write(&mut self, write: bool) {
        self.write = write;
    }

    pub(crate) fn append(&mut self, append: bool) {
        self.append = append;
    }

    pub(crate) fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }

    pub(crate) fn create(&mut self, create: bool) {
        self.create = create;
    }

    pub(crate) fn create_new(&mut self, create_new: bool) {
        self.create_new = create_new;
    }
}

/// Inner content of [`crate::fs::File`]
pub struct File(pub Box<dyn FileApi>);

/// Object-oriented manipulation of a [`File`]
pub trait FileApi: fmt::Debug {
    fn file_attr(&self) -> io::Result<FileAttr>;
    fn fsync(&self) -> io::Result<()>;
    fn datasync(&self) -> io::Result<()>;
    fn truncate(&self, _size: u64) -> io::Result<()>;
    fn read(&self, _buf: &mut [u8]) -> io::Result<usize>;
    fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize>;
    fn is_read_vectored(&self) -> bool;
    fn read_buf(&self, _cursor: BorrowedCursor<'_>) -> io::Result<()>;
    fn write(&self, _buf: &[u8]) -> io::Result<usize>;
    fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize>;
    fn is_write_vectored(&self) -> bool;
    fn flush(&self) -> io::Result<()>;
    fn seek(&self, _pos: SeekFrom) -> io::Result<u64>;
    fn duplicate(&self) -> io::Result<File>;
    fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()>;
    fn set_times(&self, _times: FileTimes) -> io::Result<()>;
}

impl File {
    pub(crate) fn open(p: &Path, opts: &OpenOptions) -> io::Result<File> {
        custom_os_impl!(fs, open, p, opts)
    }
}

impl core::ops::Deref for File {
    type Target = dyn FileApi;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Inner content of [`crate::fs::DirBuilder`]
#[derive(Debug)]
pub struct DirBuilder;

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        custom_os_impl!(fs, mkdir, p)
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    custom_os_impl!(fs, read_dir, p)
}

pub fn unlink(p: &Path) -> io::Result<()> {
    custom_os_impl!(fs, unlink, p)
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    custom_os_impl!(fs, rename, old, new)
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    custom_os_impl!(fs, set_perm, p, perm)
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    custom_os_impl!(fs, rmdir, p)
}

pub fn remove_dir_all(p: &Path) -> io::Result<()> {
    custom_os_impl!(fs, remove_dir_all, p)
}

pub fn try_exists(p: &Path) -> io::Result<bool> {
    custom_os_impl!(fs, try_exists, p)
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    custom_os_impl!(fs, readlink, p)
}

pub fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    custom_os_impl!(fs, symlink, original, link)
}

pub fn link(src: &Path, dst: &Path) -> io::Result<()> {
    custom_os_impl!(fs, link, src, dst)
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    custom_os_impl!(fs, stat, p)
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    custom_os_impl!(fs, lstat, p)
}

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    custom_os_impl!(fs, canonicalize, p)
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    custom_os_impl!(fs, copy, from, to)
}
