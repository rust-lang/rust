use crate::ffi::{CStr, OsString};
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::io::{self, Error, ErrorKind};
use crate::io::{BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::cvt;
use crate::sys::hermit::abi;
use crate::sys::hermit::abi::{O_APPEND, O_CREAT, O_EXCL, O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY};
use crate::sys::hermit::fd::FileDesc;
use crate::sys::time::SystemTime;
use crate::sys::unsupported;

pub use crate::sys_common::fs::{copy, try_exists};
//pub use crate::sys_common::fs::remove_dir_all;

#[derive(Debug)]
pub struct File(FileDesc);

pub struct FileAttr(!);

pub struct ReadDir(!);

pub struct DirEntry(!);

#[derive(Clone, Debug)]
pub struct OpenOptions {
    // generic
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    // system-specific
    mode: i32,
}

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
        OpenOptions {
            // generic
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            // system-specific
            mode: 0x777,
        }
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }
    pub fn write(&mut self, write: bool) {
        self.write = write;
    }
    pub fn append(&mut self, append: bool) {
        self.append = append;
    }
    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }
    pub fn create(&mut self, create: bool) {
        self.create = create;
    }
    pub fn create_new(&mut self, create_new: bool) {
        self.create_new = create_new;
    }

    fn get_access_mode(&self) -> io::Result<i32> {
        match (self.read, self.write, self.append) {
            (true, false, false) => Ok(O_RDONLY),
            (false, true, false) => Ok(O_WRONLY),
            (true, true, false) => Ok(O_RDWR),
            (false, _, true) => Ok(O_WRONLY | O_APPEND),
            (true, _, true) => Ok(O_RDWR | O_APPEND),
            (false, false, false) => {
                Err(io::const_io_error!(ErrorKind::InvalidInput, "invalid access mode"))
            }
        }
    }

    fn get_creation_mode(&self) -> io::Result<i32> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::const_io_error!(
                        ErrorKind::InvalidInput,
                        "invalid creation mode",
                    ));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::const_io_error!(
                        ErrorKind::InvalidInput,
                        "invalid creation mode",
                    ));
                }
            }
        }

        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => 0,
            (true, false, false) => O_CREAT,
            (false, true, false) => O_TRUNC,
            (true, true, false) => O_CREAT | O_TRUNC,
            (_, _, true) => O_CREAT | O_EXCL,
        })
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        run_path_with_cstr(path, |path| File::open_c(&path, opts))
    }

    pub fn open_c(path: &CStr, opts: &OpenOptions) -> io::Result<File> {
        let mut flags = opts.get_access_mode()?;
        flags = flags | opts.get_creation_mode()?;

        let mode;
        if flags & O_CREAT == O_CREAT {
            mode = opts.mode;
        } else {
            mode = 0;
        }

        let fd = unsafe { cvt(abi::open(path.as_ptr(), flags, mode))? };
        Ok(File(FileDesc::new(fd as i32)))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn fsync(&self) -> io::Result<()> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, _pos: SeekFrom) -> io::Result<u64> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn duplicate(&self) -> io::Result<File> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        Err(Error::from_raw_os_error(22))
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

pub fn readdir(_p: &Path) -> io::Result<ReadDir> {
    unsupported()
}

pub fn unlink(path: &Path) -> io::Result<()> {
    run_path_with_cstr(path, |path| cvt(unsafe { abi::unlink(path.as_ptr()) }).map(|_| ()))
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
    //unsupported()
    Ok(())
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn symlink(_original: &Path, _link: &Path) -> io::Result<()> {
    unsupported()
}

pub fn link(_original: &Path, _link: &Path) -> io::Result<()> {
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
