use super::abi::{self, O_APPEND, O_CREAT, O_EXCL, O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY};
use super::fd::FileDesc;
use crate::ffi::{CStr, OsString};
use crate::fmt;
use crate::io::{self, Error, ErrorKind};
use crate::io::{BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::os::hermit::ffi::OsStringExt;
use crate::os::hermit::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::sync::Arc;
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::cvt;
use crate::sys::time::SystemTime;
use crate::sys::unsupported;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

pub use crate::sys_common::fs::{copy, try_exists};
//pub use crate::sys_common::fs::remove_dir_all;

#[derive(Debug)]
pub struct File(FileDesc);
#[derive(Clone)]
pub struct FileAttr {
    stat_val: stat_struct,
}

impl FileAttr {
    fn from_stat(stat_val: stat_struct) -> Self {
        Self { stat_val }
    }
}

// all DirEntry's will have a reference to this struct
struct InnerReadDir {
    dirp: FileDesc,
    root: PathBuf,
}

pub struct ReadDir {
    inner: Arc<InnerReadDir>,
    end_of_stream: bool,
}

impl ReadDir {
    fn new(inner: InnerReadDir) -> Self {
        Self { inner: Arc::new(inner), end_of_stream: false }
    }
}

pub struct DirEntry {
    dir: Arc<InnerReadDir>,
    entry: dirent_min,
    name: OsString,
}

struct dirent_min {
    d_ino: u64,
    d_type: u32,
}

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

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    mode: u32,
}

#[derive(Copy, Clone, Eq, Debug)]
pub struct FileType {
    mode: u32,
}

impl PartialEq for FileType {
    fn eq(&self, other: &Self) -> bool {
        self.mode == other.mode
    }
}

impl core::hash::Hash for FileType {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.mode.hash(state);
    }
}

#[derive(Debug)]
pub struct DirBuilder {
    mode: u32,
}

impl FileAttr {
    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::new(self.stat_val.st_mtime, self.stat_val.st_mtime_nsec))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::new(self.stat_val.st_atime, self.stat_val.st_atime_nsec))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::new(self.stat_val.st_ctime, self.stat_val.st_ctime_nsec))
    }

    pub fn size(&self) -> u64 {
        self.stat_val.st_size as u64
    }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: (self.stat_val.st_mode) }
    }

    pub fn file_type(&self) -> FileType {
        let masked_mode = self.stat_val.st_mode & S_IFMT;
        let mode = match masked_mode {
            S_IFDIR => DT_DIR,
            S_IFLNK => DT_LNK,
            S_IFREG => DT_REG,
            _ => DT_UNKNOWN,
        };
        FileType { mode: mode }
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        // check if any class (owner, group, others) has write permission
        self.mode & 0o222 == 0
    }

    pub fn set_readonly(&mut self, _readonly: bool) {
        unimplemented!()
    }

    #[allow(dead_code)]
    pub fn mode(&self) -> u32 {
        self.mode as u32
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, _t: SystemTime) {}
    pub fn set_modified(&mut self, _t: SystemTime) {}
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.mode == DT_DIR
    }
    pub fn is_file(&self) -> bool {
        self.mode == DT_REG
    }
    pub fn is_symlink(&self) -> bool {
        self.mode == DT_LNK
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This will only be called from std::fs::ReadDir, which will add a "ReadDir()" frame.
        // Thus the result will be e g 'ReadDir("/home")'
        fmt::Debug::fmt(&*self.inner.root, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        if self.end_of_stream {
            return None;
        }

        unsafe {
            loop {
                // As of POSIX.1-2017, readdir() is not required to be thread safe; only
                // readdir_r() is. However, readdir_r() cannot correctly handle platforms
                // with unlimited or variable NAME_MAX. Many modern platforms guarantee
                // thread safety for readdir() as long an individual DIR* is not accessed
                // concurrently, which is sufficient for Rust.
                let entry_ptr = match abi::readdir(self.inner.dirp.as_raw_fd()) {
                    abi::DirectoryEntry::Invalid(e) => {
                        // We either encountered an error, or reached the end. Either way,
                        // the next call to next() should return None.
                        self.end_of_stream = true;

                        return Some(Err(Error::from_raw_os_error(e)));
                    }
                    abi::DirectoryEntry::Valid(ptr) => {
                        if ptr.is_null() {
                            return None;
                        }

                        ptr
                    }
                };

                macro_rules! offset_ptr {
                    ($entry_ptr:expr, $field:ident) => {{
                        const OFFSET: isize = {
                            let delusion = MaybeUninit::<dirent>::uninit();
                            let entry_ptr = delusion.as_ptr();
                            unsafe {
                                ptr::addr_of!((*entry_ptr).$field)
                                    .cast::<u8>()
                                    .offset_from(entry_ptr.cast::<u8>())
                            }
                        };
                        if true {
                            // Cast to the same type determined by the else branch.
                            $entry_ptr.byte_offset(OFFSET).cast::<_>()
                        } else {
                            #[allow(deref_nullptr)]
                            {
                                ptr::addr_of!((*ptr::null::<dirent>()).$field)
                            }
                        }
                    }};
                }

                // d_name is NOT guaranteed to be null-terminated.
                let name_bytes = core::slice::from_raw_parts(
                    offset_ptr!(entry_ptr, d_name) as *const u8,
                    *offset_ptr!(entry_ptr, d_namelen) as usize,
                )
                .to_vec();

                if name_bytes == b"." || name_bytes == b".." {
                    continue;
                }

                let name = OsString::from_vec(name_bytes);

                let entry = dirent_min {
                    d_ino: *offset_ptr!(entry_ptr, d_ino),
                    d_type: *offset_ptr!(entry_ptr, d_type),
                };

                return Some(Ok(DirEntry { entry, name: name, dir: Arc::clone(&self.inner) }));
            }
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.dir.root.join(self.file_name_os_str())
    }

    pub fn file_name(&self) -> OsString {
        self.file_name_os_str().to_os_string()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        lstat(&self.path())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType { mode: self.entry.d_type })
    }

    #[allow(dead_code)]
    pub fn ino(&self) -> u64 {
        self.entry.d_ino
    }

    pub fn file_name_os_str(&self) -> &OsStr {
        self.name.as_os_str()
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
            mode: 0o777,
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
        run_path_with_cstr(path, &|path| File::open_c(&path, opts))
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
        Ok(File(unsafe { FileDesc::from_raw_fd(fd as i32) }))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let mut stat_val: stat_struct = unsafe { mem::zeroed() };
        self.0.fstat(&mut stat_val)?;
        Ok(FileAttr::from_stat(stat_val))
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

    #[inline]
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
        DirBuilder { mode: 0o777 }
    }

    pub fn mkdir(&self, path: &Path) -> io::Result<()> {
        run_path_with_cstr(path, |path| {
            cvt(unsafe { abi::mkdir(path.as_ptr(), self.mode) }).map(|_| ())
        })
    }

    #[allow(dead_code)]
    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode as u32;
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
        Self(FromRawFd::from_raw_fd(raw_fd))
    }
}

pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    let fd_raw = run_path_with_cstr(path, |path| cvt(unsafe { abi::opendir(path.as_ptr()) }))?;
    let fd = unsafe { FileDesc::from_raw_fd(fd_raw as i32) };
    let root = path.to_path_buf();
    let inner = InnerReadDir { dirp: fd, root };
    Ok(ReadDir::new(inner))
}

pub fn unlink(path: &Path) -> io::Result<()> {
    run_path_with_cstr(path, &|path| cvt(unsafe { abi::unlink(path.as_ptr()) }).map(|_| ()))
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    Err(Error::from_raw_os_error(22))
}

pub fn rmdir(path: &Path) -> io::Result<()> {
    run_path_with_cstr(path, |path| cvt(unsafe { abi::rmdir(path.as_ptr()) }).map(|_| ()))
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

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    run_path_with_cstr(path, |path| {
        let mut stat_val: stat_struct = unsafe { mem::zeroed() };
        cvt(unsafe { abi::stat(path.as_ptr(), &mut stat_val) })?;
        Ok(FileAttr::from_stat(stat_val))
    })
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    run_path_with_cstr(path, |path| {
        let mut stat_val: stat_struct = unsafe { mem::zeroed() };
        cvt(unsafe { abi::lstat(path.as_ptr(), &mut stat_val) })?;
        Ok(FileAttr::from_stat(stat_val))
    })
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}
