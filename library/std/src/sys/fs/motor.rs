use crate::ffi::OsString;
use crate::hash::Hash;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::path::{Path, PathBuf};
use crate::sys::fd::FileDesc;
pub use crate::sys::fs::common::exists;
use crate::sys::time::SystemTime;
use crate::sys::{map_motor_error, unsupported};
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileType {
    rt_filetype: u8,
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.rt_filetype == moto_rt::fs::FILETYPE_DIRECTORY
    }

    pub fn is_file(&self) -> bool {
        self.rt_filetype == moto_rt::fs::FILETYPE_FILE
    }

    pub fn is_symlink(&self) -> bool {
        false
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FilePermissions {
    rt_perm: u64,
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        (self.rt_perm & moto_rt::fs::PERM_WRITE == 0)
            && (self.rt_perm & moto_rt::fs::PERM_READ != 0)
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.rt_perm = moto_rt::fs::PERM_READ;
        } else {
            self.rt_perm = moto_rt::fs::PERM_READ | moto_rt::fs::PERM_WRITE;
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    modified: u128,
    accessed: u128,
}

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = t.as_u128();
    }

    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = t.as_u128();
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct FileAttr {
    inner: moto_rt::fs::FileAttr,
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { rt_perm: self.inner.perm }
    }

    pub fn file_type(&self) -> FileType {
        FileType { rt_filetype: self.inner.file_type }
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        match self.inner.modified {
            0 => Err(crate::io::Error::from(crate::io::ErrorKind::Other)),
            x => Ok(SystemTime::from_u128(x)),
        }
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        match self.inner.accessed {
            0 => Err(crate::io::Error::from(crate::io::ErrorKind::Other)),
            x => Ok(SystemTime::from_u128(x)),
        }
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        match self.inner.created {
            0 => Err(crate::io::Error::from(crate::io::ErrorKind::Other)),
            x => Ok(SystemTime::from_u128(x)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    rt_open_options: u32,
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions { rt_open_options: 0 }
    }

    pub fn read(&mut self, read: bool) {
        if read {
            self.rt_open_options |= moto_rt::fs::O_READ;
        } else {
            self.rt_open_options &= !moto_rt::fs::O_READ;
        }
    }

    pub fn write(&mut self, write: bool) {
        if write {
            self.rt_open_options |= moto_rt::fs::O_WRITE;
        } else {
            self.rt_open_options &= !moto_rt::fs::O_WRITE;
        }
    }

    pub fn append(&mut self, append: bool) {
        if append {
            self.rt_open_options |= moto_rt::fs::O_APPEND;
        } else {
            self.rt_open_options &= !moto_rt::fs::O_APPEND;
        }
    }

    pub fn truncate(&mut self, truncate: bool) {
        if truncate {
            self.rt_open_options |= moto_rt::fs::O_TRUNCATE;
        } else {
            self.rt_open_options &= !moto_rt::fs::O_TRUNCATE;
        }
    }

    pub fn create(&mut self, create: bool) {
        if create {
            self.rt_open_options |= moto_rt::fs::O_CREATE;
        } else {
            self.rt_open_options &= !moto_rt::fs::O_CREATE;
        }
    }

    pub fn create_new(&mut self, create_new: bool) {
        if create_new {
            self.rt_open_options |= moto_rt::fs::O_CREATE_NEW;
        } else {
            self.rt_open_options &= !moto_rt::fs::O_CREATE_NEW;
        }
    }
}

#[derive(Debug)]
pub struct File(FileDesc);

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
        moto_rt::fs::open(path, opts.rt_open_options)
            .map(|fd| unsafe { Self::from_raw_fd(fd) })
            .map_err(map_motor_error)
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        moto_rt::fs::get_file_attr(self.as_raw_fd())
            .map(|inner| -> FileAttr { FileAttr { inner } })
            .map_err(map_motor_error)
    }

    pub fn fsync(&self) -> io::Result<()> {
        moto_rt::fs::fsync(self.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn datasync(&self) -> io::Result<()> {
        moto_rt::fs::datasync(self.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        moto_rt::fs::truncate(self.as_raw_fd(), size).map_err(map_motor_error)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        moto_rt::fs::read(self.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|b| self.read(b), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        moto_rt::fs::write(self.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        moto_rt::fs::flush(self.as_raw_fd()).map_err(map_motor_error)
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                moto_rt::fs::seek(self.as_raw_fd(), offset as i64, moto_rt::fs::SEEK_SET)
                    .map_err(map_motor_error)
            }
            SeekFrom::End(offset) => {
                moto_rt::fs::seek(self.as_raw_fd(), offset, moto_rt::fs::SEEK_END)
                    .map_err(map_motor_error)
            }
            SeekFrom::Current(offset) => {
                moto_rt::fs::seek(self.as_raw_fd(), offset, moto_rt::fs::SEEK_CUR)
                    .map_err(map_motor_error)
            }
        }
    }

    pub fn tell(&self) -> io::Result<u64> {
        self.seek(SeekFrom::Current(0))
    }

    pub fn duplicate(&self) -> io::Result<File> {
        moto_rt::fs::duplicate(self.as_raw_fd())
            .map(|fd| unsafe { Self::from_raw_fd(fd) })
            .map_err(map_motor_error)
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        moto_rt::fs::set_file_perm(self.as_raw_fd(), perm.rt_perm).map_err(map_motor_error)
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        unsupported() // Let's not do that.
    }

    pub fn lock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn try_lock(&self) -> Result<(), crate::fs::TryLockError> {
        Err(crate::fs::TryLockError::Error(io::Error::from(io::ErrorKind::Unsupported)))
    }

    pub fn try_lock_shared(&self) -> Result<(), crate::fs::TryLockError> {
        Err(crate::fs::TryLockError::Error(io::Error::from(io::ErrorKind::Unsupported)))
    }

    pub fn unlock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn size(&self) -> Option<io::Result<u64>> {
        None
    }
}

#[derive(Debug)]
pub struct DirBuilder {}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, path: &Path) -> io::Result<()> {
        let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
        moto_rt::fs::mkdir(path).map_err(map_motor_error)
    }
}

pub fn unlink(path: &Path) -> io::Result<()> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    moto_rt::fs::unlink(path).map_err(map_motor_error)
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let old = old.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    let new = new.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    moto_rt::fs::rename(old, new).map_err(map_motor_error)
}

pub fn rmdir(path: &Path) -> io::Result<()> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    moto_rt::fs::rmdir(path).map_err(map_motor_error)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    moto_rt::fs::rmdir_all(path).map_err(map_motor_error)
}

pub fn set_perm(path: &Path, perm: FilePermissions) -> io::Result<()> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    moto_rt::fs::set_perm(path, perm.rt_perm).map_err(map_motor_error)
}

pub fn set_times(_p: &Path, _times: FileTimes) -> io::Result<()> {
    unsupported()
}

pub fn set_times_nofollow(_p: &Path, _times: FileTimes) -> io::Result<()> {
    unsupported()
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn symlink(_original: &Path, _link: &Path) -> io::Result<()> {
    unsupported()
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    let inner = moto_rt::fs::stat(path).map_err(map_motor_error)?;
    Ok(FileAttr { inner })
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    stat(path)
}

pub fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    let path = moto_rt::fs::canonicalize(path).map_err(map_motor_error)?;
    Ok(path.into())
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    let from = from.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    let to = to.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    moto_rt::fs::copy(from, to).map_err(map_motor_error)
}

#[derive(Debug)]
pub struct ReadDir {
    rt_fd: moto_rt::RtFd,
    path: String,
}

impl Drop for ReadDir {
    fn drop(&mut self) {
        moto_rt::fs::closedir(self.rt_fd).unwrap();
    }
}

pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    Ok(ReadDir {
        rt_fd: moto_rt::fs::opendir(path).map_err(map_motor_error)?,
        path: path.to_owned(),
    })
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        match moto_rt::fs::readdir(self.rt_fd).map_err(map_motor_error) {
            Ok(maybe_item) => match maybe_item {
                Some(inner) => Some(Ok(DirEntry { inner, parent_path: self.path.clone() })),
                None => None,
            },
            Err(err) => Some(Err(err)),
        }
    }
}

pub struct DirEntry {
    parent_path: String,
    inner: moto_rt::fs::DirEntry,
}

impl DirEntry {
    fn filename(&self) -> &str {
        core::str::from_utf8(unsafe {
            core::slice::from_raw_parts(self.inner.fname.as_ptr(), self.inner.fname_size as usize)
        })
        .unwrap()
    }

    pub fn path(&self) -> PathBuf {
        let mut path = self.parent_path.clone();
        path.push_str("/");
        path.push_str(self.filename());
        path.into()
    }

    pub fn file_name(&self) -> OsString {
        self.filename().to_owned().into()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        Ok(FileAttr { inner: self.inner.attr })
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType { rt_filetype: self.inner.attr.file_type })
    }
}

impl AsInner<FileDesc> for File {
    #[inline]
    fn as_inner(&self) -> &FileDesc {
        &self.0
    }
}

impl AsInnerMut<FileDesc> for File {
    #[inline]
    fn as_inner_mut(&mut self) -> &mut FileDesc {
        &mut self.0
    }
}

impl IntoInner<FileDesc> for File {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

impl FromInner<FileDesc> for File {
    fn from_inner(file_desc: FileDesc) -> Self {
        Self(file_desc)
    }
}

impl AsFd for File {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl AsRawFd for File {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl IntoRawFd for File {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for File {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self(FromRawFd::from_raw_fd(raw_fd)) }
    }
}
