use crate::ffi::{CStr, OsStr, OsString};
use crate::fs::TryLockError;
use crate::io::{self, BorrowedCursor, Error, ErrorKind, IoSlice, IoSliceMut, SeekFrom};
use crate::mem::MaybeUninit;
use crate::os::hermit::ffi::OsStringExt;
use crate::os::hermit::hermit_abi::{
    self, DT_DIR, DT_LNK, DT_REG, DT_UNKNOWN, O_APPEND, O_CREAT, O_DIRECTORY, O_EXCL, O_RDONLY,
    O_RDWR, O_TRUNC, O_WRONLY, S_IFDIR, S_IFLNK, S_IFMT, S_IFREG, dirent64, stat as stat_struct,
};
use crate::os::hermit::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::fd::FileDesc;
pub use crate::sys::fs::common::{Dir, copy, exists};
use crate::sys::helpers::run_path_with_cstr;
use crate::sys::io::DEFAULT_BUF_SIZE;
use crate::sys::time::SystemTime;
use crate::sys::{AsInner, AsInnerMut, FromInner, IntoInner, cvt, unsupported, unsupported_err};
use crate::{cmp, fmt, mem, slice};

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
    root: PathBuf,
}

pub struct ReadDir {
    inner: Arc<InnerReadDir>,
    fd: FileDesc,
    buf: GetdentsBuffer,
}

/// A buffer containing [`dirent64`]s, filled with [`getdents64`].
///
/// This struct is roughly modeled after the `BufReader`'s `Buffer`.
struct GetdentsBuffer {
    // The buffer.
    buf: Box<[MaybeUninit<dirent64>]>,
    // The current seek offset into `buf`, must always be <= `filled`.
    pos: usize,
    // Each call to `fill_buf` sets `filled` to indicate how many bytes at the start of `buf` are
    // initialized with bytes from a read.
    filled: usize,
}

impl GetdentsBuffer {
    /// Creates a new buffer with at least `capacity` bytes for use with dirent.
    fn with_capacity(capacity: usize) -> Self {
        let buf = Box::new_uninit_slice(capacity.div_ceil(size_of::<dirent64>()));
        Self { buf, pos: 0, filled: 0 }
    }

    fn buffer(&self) -> &[u8] {
        // SAFETY: self.pos and self.filled are valid, and self.filled >= self.pos, and
        // that region is initialized because those are all invariants of this type.
        unsafe {
            let ptr = self.buf.as_ptr().cast::<MaybeUninit<u8>>().add(self.pos);
            slice::from_raw_parts(ptr, self.filled - self.pos).assume_init_ref()
        }
    }

    fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.filled);
    }

    fn fill_buf(&mut self, fd: BorrowedFd<'_>) -> io::Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.filled {
            debug_assert!(self.pos == self.filled);

            let result = unsafe {
                cvt(hermit_abi::getdents64(
                    fd.as_raw_fd(),
                    self.buf.as_mut_ptr().cast(),
                    self.buf.len() * size_of::<dirent64>(),
                ))
            };

            self.pos = 0;
            self.filled = 0;

            self.filled = result? as usize;
        }

        Ok(self.buffer())
    }
}

pub struct DirEntry {
    dir: Arc<InnerReadDir>,
    /// 64-bit inode number
    ino: u64,
    /// File type
    type_: u8,
    /// name of the entry
    name: OsString,
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
    mode: u8,
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
        SystemTime::new(self.stat_val.st_mtim.tv_sec, self.stat_val.st_mtim.tv_nsec.into())
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat_val.st_atim.tv_sec, self.stat_val.st_atim.tv_nsec.into())
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat_val.st_ctim.tv_sec, self.stat_val.st_ctim.tv_nsec.into())
    }

    pub fn size(&self) -> u64 {
        self.stat_val.st_size as u64
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: self.stat_val.st_mode }
    }

    pub fn file_type(&self) -> FileType {
        let masked_mode = self.stat_val.st_mode & S_IFMT;
        let mode = match masked_mode {
            S_IFDIR => DT_DIR,
            S_IFLNK => DT_LNK,
            S_IFREG => DT_REG,
            _ => DT_UNKNOWN,
        };
        FileType { mode }
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
        // Thus the result will be e.g. 'ReadDir("/home")'
        fmt::Debug::fmt(&*self.inner.root, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        loop {
            let buf = match self.buf.fill_buf(self.fd.as_fd()) {
                Ok(buf) => buf,
                Err(err) => return Some(Err(err)),
            };

            if buf.len() == 0 {
                // No more entries left.
                return None;
            }

            let entry_ptr = buf.as_ptr().cast::<dirent64>();

            // The dirent64 struct is a weird imaginary thing that isn't ever supposed
            // to be worked with by value. Its trailing d_name field is declared
            // variously as [c_char; 256] or [c_char; 1] on different systems but
            // either way that size is meaningless; only the offset of d_name is
            // meaningful. The dirent64 pointers that libc returns from getdents64 are
            // allowed to point to allocations smaller _or_ LARGER than implied by the
            // definition of the struct.
            //
            // As such, we need to be even more careful with dirent64 than if its
            // contents were "simply" partially initialized data.
            //
            // Like for uninitialized contents, converting entry_ptr to `&dirent64`
            // would not be legal. However, we can use `&raw const (*entry_ptr).d_name`
            // to refer the fields individually, because that operation is equivalent
            // to `byte_offset` and thus does not require the full extent of `*entry_ptr`
            // to be in bounds of the same allocation, only the offset of the field
            // being referenced.

            self.buf.consume(usize::from(unsafe { (*entry_ptr).d_reclen }));

            // d_name is guaranteed to be null-terminated.
            let name = unsafe { CStr::from_ptr((&raw const (*entry_ptr).d_name).cast()) };
            let name_bytes = name.to_bytes();
            if name_bytes == b"." || name_bytes == b".." {
                continue;
            }

            return Some(Ok(DirEntry {
                dir: Arc::clone(&self.inner),
                ino: unsafe { (*entry_ptr).d_ino },
                type_: unsafe { (*entry_ptr).d_type },
                name: OsString::from_vec(name_bytes.to_vec()),
            }));
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
        let mut path = self.path();
        path.set_file_name(self.file_name_os_str());
        lstat(&path)
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType { mode: self.type_ })
    }

    #[allow(dead_code)]
    pub fn ino(&self) -> u64 {
        self.ino
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
                Err(io::const_error!(ErrorKind::InvalidInput, "invalid access mode"))
            }
        }
    }

    fn get_creation_mode(&self) -> io::Result<i32> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::const_error!(ErrorKind::InvalidInput, "invalid creation mode"));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::const_error!(ErrorKind::InvalidInput, "invalid creation mode"));
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

        let fd = unsafe { cvt(hermit_abi::open(path.as_ptr(), flags, mode))? };
        Ok(File(unsafe { FileDesc::from_raw_fd(fd) }))
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

    pub fn lock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn try_lock(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(unsupported_err()))
    }

    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(unsupported_err()))
    }

    pub fn unlock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        Err(Error::from_raw_os_error(22))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_, u8>) -> io::Result<()> {
        self.0.read_buf(cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    #[inline]
    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        self.0.seek(pos)
    }

    pub fn size(&self) -> Option<io::Result<u64>> {
        None
    }

    pub fn tell(&self) -> io::Result<u64> {
        self.0.tell()
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
        run_path_with_cstr(path, &|path| {
            cvt(unsafe { hermit_abi::mkdir(path.as_ptr().cast(), self.mode.into()) }).map(|_| ())
        })
    }

    #[allow(dead_code)]
    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode;
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
        let file_desc = unsafe { FileDesc::from_raw_fd(raw_fd) };
        Self(file_desc)
    }
}

pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    let fd_raw = run_path_with_cstr(path, &|path| {
        cvt(unsafe { hermit_abi::open(path.as_ptr(), O_RDONLY | O_DIRECTORY, 0) })
    })?;
    let fd = unsafe { FileDesc::from_raw_fd(fd_raw) };

    let root = path.to_path_buf();
    let inner = Arc::new(InnerReadDir { root });
    let buf_size = usize::max(DEFAULT_BUF_SIZE, size_of::<dirent64>());
    let buf = GetdentsBuffer::with_capacity(buf_size);

    Ok(ReadDir { inner, fd, buf })
}

pub fn unlink(path: &Path) -> io::Result<()> {
    run_path_with_cstr(path, &|path| cvt(unsafe { hermit_abi::unlink(path.as_ptr()) }).map(|_| ()))
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    Err(Error::from_raw_os_error(22))
}

pub fn set_times(_p: &Path, _times: FileTimes) -> io::Result<()> {
    Err(Error::from_raw_os_error(22))
}

pub fn set_times_nofollow(_p: &Path, _times: FileTimes) -> io::Result<()> {
    Err(Error::from_raw_os_error(22))
}

pub fn rmdir(path: &Path) -> io::Result<()> {
    run_path_with_cstr(path, &|path| cvt(unsafe { hermit_abi::rmdir(path.as_ptr()) }).map(|_| ()))
}

pub fn remove_dir_all(_path: &Path) -> io::Result<()> {
    unsupported()
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
    run_path_with_cstr(path, &|path| {
        let mut stat_val: stat_struct = unsafe { mem::zeroed() };
        cvt(unsafe { hermit_abi::stat(path.as_ptr(), &mut stat_val) })?;
        Ok(FileAttr::from_stat(stat_val))
    })
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    run_path_with_cstr(path, &|path| {
        let mut stat_val: stat_struct = unsafe { mem::zeroed() };
        cvt(unsafe { hermit_abi::lstat(path.as_ptr(), &mut stat_val) })?;
        Ok(FileAttr::from_stat(stat_val))
    })
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}
