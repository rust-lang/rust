use crate::ffi::{CString, OsString};
use crate::fmt;
use crate::hash::Hash;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;

struct FileDesc(*mut vex_sdk::FIL);

pub struct File(FileDesc);

//TODO: We may be able to get some of this info
#[derive(Clone)]
pub struct FileAttr;

pub struct ReadDir(!);

pub struct DirEntry(!);

#[derive(Clone, Debug)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilePermissions;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct FileType {
    is_dir: bool,
}

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        todo!()
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions
    }

    pub fn file_type(&self) -> FileType {
        todo!()
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        todo!()
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        todo!()
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        todo!()
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        false
    }

    pub fn set_readonly(&mut self, _readonly: bool) {
        panic!("Perimissions do not exist")
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, _t: SystemTime) {}
    pub fn set_modified(&mut self, _t: SystemTime) {}
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.is_dir
    }

    pub fn is_file(&self) -> bool {
        !self.is_dir
    }

    pub fn is_symlink(&self) -> bool {
        false
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
        OpenOptions { read: false, write: false, append: false }
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
    pub fn truncate(&mut self, _truncate: bool) {}
    pub fn create(&mut self, create: bool) {
        self.write = create;
    }
    pub fn create_new(&mut self, _create_new: bool) {}
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let fs_status = unsafe { vex_sdk::vexFileMountSD() };
        match fs_status {
            vex_sdk::FRESULT::FR_OK => (),
            //TODO: cover more results
            _ => return Err(io::Error::new(io::ErrorKind::NotFound, "SD card cannot be written or read from")),
        }

        let path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Path contained a null byte")
        })?;

        let file = if opts.read && !opts.write {
            // The second argument to this function is ignored.
            // Open in read only mode
            unsafe { vex_sdk::vexFileOpen(path.as_ptr(), c"".as_ptr()) }
        } else if opts.write && opts.append {
            // Open in read/write and append mode
            unsafe { vex_sdk::vexFileOpenWrite(path.as_ptr()) }
        } else if opts.write {
            // Open in read/write mode
            unsafe { vex_sdk::vexFileOpenCreate(path.as_ptr()) }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Files cannot be opened without read or write access",
            ));
        };

        if file.is_null() {
            Err(io::Error::new(io::ErrorKind::NotFound, "Could not open file"))
        } else {
            Ok(Self(FileDesc(file)))
        }
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        todo!()
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        unsupported()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len() as _;
        let buf_ptr = buf.as_mut_ptr();
        let read = unsafe { vex_sdk::vexFileRead(buf_ptr.cast(), 1, len, self.0.0) };
        if read < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Could not read from file"))
        } else {
            Ok(read as usize)
        }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|b| self.read(b), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|b| self.read(b), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        let buf_ptr = buf.as_ptr();
        let written = unsafe { vex_sdk::vexFileWrite(buf_ptr.cast_mut().cast(), 1, len as _, self.0.0) };
        if written < 0 {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Could not write to file",
            ))
        } else {
            Ok(written as usize)
        }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        unsafe {
            vex_sdk::vexFileSync(self.0.0);
        }
        Ok(())
    }

    pub fn seek(&self, _pos: SeekFrom) -> io::Result<u64> {
        todo!();
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unsupported!()
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        unsupported()
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        unsupported()
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").finish_non_exhaustive()
    }
}
impl Drop for File {
    fn drop(&mut self) {
        unsafe { vex_sdk::vexFileClose(self.0.0) };
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

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    unsupported()
}

pub fn rmdir(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn remove_dir_all(_path: &Path) -> io::Result<()> {
    unsupported()
}

pub fn try_exists(_path: &Path) -> io::Result<bool> {
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
