//! ThingOS PAL — filesystem support.
//!
//! This module provides the platform-specific filesystem implementation for
//! `std::fs` on ThingOS.  It maps directly to the VFS syscalls exposed by
//! the ThingOS kernel (see `abi/src/numbers.rs`).
//!
//! # Symlink support
//! ThingOS VFS has basic symlink support (`SYS_FS_SYMLINK`, `SYS_FS_READLINK`,
//! `SYS_FS_LSTAT`).  `stat` opens the path via `SYS_FS_OPEN` (which follows
//! symlinks) and then calls `SYS_FS_STAT` on the resulting fd.  `lstat` uses
//! `SYS_FS_LSTAT`, a dedicated path-based syscall that resolves the final
//! path component without following it, returning symlink metadata directly.
//! Hard links are supported via `SYS_FS_LINK` for regular files on ramfs
//! mount points.
//!
//! # Permissions
//! ThingOS exposes POSIX-style `mode` bits in the stat structure.  Permissions
//! can be changed via `SYS_FS_CHMOD` (path-based) and `SYS_FS_FCHMOD` (fd-based).
//!
//! # Timestamps
//! atime/mtime/ctime can be updated via `SYS_FS_UTIMES` (path-based) and
//! `SYS_FS_FUTIMES` (fd-based).  Pass `u64::MAX` for `atime_sec` or `mtime_sec`
//! in the request struct to leave that timestamp unchanged.

use crate::ffi::OsString;
use crate::fs::TryLockError;
use crate::hash::{Hash, Hasher};
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::time::SystemTime;
pub use crate::sys::fs::common::{Dir, copy, exists, remove_dir_all};
use crate::{fmt, vec};

use crate::sys::pal::raw_syscall6;

// ── Syscall numbers (abi/src/numbers.rs) ─────────────────────────────────────
const SYS_FS_OPEN: u32 = 0x4000;
const SYS_FS_CLOSE: u32 = 0x4001;
const SYS_FS_READ: u32 = 0x4002;
const SYS_FS_WRITE: u32 = 0x4003;
const SYS_FS_SEEK: u32 = 0x4004;
const SYS_FS_STAT: u32 = 0x4005;
const SYS_FS_READDIR: u32 = 0x4006;
const SYS_FS_MKDIR: u32 = 0x4007;
const SYS_FS_UNLINK: u32 = 0x4008;
const SYS_FS_DUP: u32 = 0x400C;
const SYS_FS_RENAME: u32 = 0x4010;
const SYS_FS_SYNC: u32 = 0x4017;
const SYS_FS_REALPATH: u32 = 0x4016;
const SYS_FS_SYMLINK: u32 = 0x4019;
const SYS_FS_READLINK: u32 = 0x401A;
const SYS_FS_FTRUNCATE: u32 = 0x401B;
const SYS_FS_CHMOD: u32 = 0x401C;
const SYS_FS_FCHMOD: u32 = 0x401D;
const SYS_FS_UTIMES: u32 = 0x401E;
const SYS_FS_FUTIMES: u32 = 0x401F;
const SYS_FS_LSTAT: u32 = 0x4020;
const SYS_FS_READV: u32 = 0x4021;
const SYS_FS_WRITEV: u32 = 0x4022;
const SYS_FS_LINK: u32 = 0x4023;
const SYS_FS_FLOCK: u32 = 0x4024;
const SYS_FS_LUTIMES: u32 = 0x4025;

// ── Advisory lock flags (mirror abi::syscall::flock_flags) ───────────────────
const LOCK_SH: u32 = 1; // shared (read) lock
const LOCK_EX: u32 = 2; // exclusive (write) lock
const LOCK_NB: u32 = 4; // non-blocking
const LOCK_UN: u32 = 8; // unlock

// ── Scatter-gather I/O vector (matches abi::syscall::IoVec / POSIX struct iovec) ──
#[repr(C)]
struct KernelIoVec {
    base: usize,
    len: usize,
}

// ── Open flags (abi/src/numbers.rs vfs_flags) ─────────────────────────────────
const O_RDONLY: u32 = 0x0000;
const O_WRONLY: u32 = 0x0001;
const O_RDWR: u32 = 0x0002;
const O_CREAT: u32 = 0x0040;
const O_EXCL: u32 = 0x0080;
const O_TRUNC: u32 = 0x0200;
const O_APPEND: u32 = 0x0400;

// ── File-type constants (POSIX st_mode bits) ──────────────────────────────────
const S_IFMT: u32 = 0o170000;
const S_IFREG: u32 = 0o100000;
const S_IFDIR: u32 = 0o040000;
const S_IFLNK: u32 = 0o120000;

// ── Seek `whence` values ──────────────────────────────────────────────────────
const SEEK_SET: u32 = 0;
const SEEK_CUR: u32 = 1;
const SEEK_END: u32 = 2;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a negative raw syscall return into an `crate::io::Error`.
#[inline]
fn syscall_err(ret: isize) -> crate::io::Error {
    crate::io::Error::from_raw_os_error((-ret) as i32)
}

/// Convert a syscall result (negative = error) into `crate::io::Result<usize>`.
#[inline]
fn cvt(ret: isize) -> crate::io::Result<usize> {
    if ret < 0 { Err(syscall_err(ret)) } else { Ok(ret as usize) }
}

/// Resolve a `Path` to a `&str` (paths are always UTF-8 on ThingOS).
#[inline]
fn path_to_str(path: &Path) -> crate::io::Result<&str> {
    path.to_str()
        .ok_or_else(|| io::const_error!(crate::io::ErrorKind::InvalidInput, "path is not valid UTF-8"))
}

// ── FileDesc: thin close-on-drop fd wrapper ───────────────────────────────────

struct FileDesc(u32);

impl FileDesc {
    fn from_raw(fd: u32) -> Self {
        Self(fd)
    }

    fn raw(&self) -> u32 {
        self.0
    }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        // SAFETY: we own this fd; closing is always valid even if the underlying
        // resource has already been released by other means.
        unsafe { raw_syscall6(SYS_FS_CLOSE, self.0 as usize, 0, 0, 0, 0, 0) };
    }
}

// ── ABI stat layout (must match abi/src/fs.rs FileStat) ──────────────────────

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct AbiTimespec {
    sec: u64,
    nsec: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct AbiFileStat {
    mode: u32,
    nlink: u32,
    size: u64,
    ino: u64,
    atime: AbiTimespec,
    mtime: AbiTimespec,
    ctime: AbiTimespec,
    uid: u32,
    gid: u32,
    rdev: u64,
    blksize: u32,
    _blksize_pad: u32,
    blocks: u64,
}

// ── ABI utimes layout (must match abi/src/fs.rs UtimesRequest) ────────────────

/// Sentinel for `atime_sec`/`mtime_sec` meaning "do not update this timestamp".
const UTIME_OMIT: u64 = u64::MAX;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct AbiUtimesRequest {
    atime_sec: u64,
    atime_nsec: u32,
    _pad1: u32,
    mtime_sec: u64,
    mtime_nsec: u32,
    _pad2: u32,
}

impl AbiUtimesRequest {
    fn from_file_times(times: &FileTimes) -> Self {
        let (atime_sec, atime_nsec) = match times.accessed {
            Some(t) => {
                let d = t.as_duration();
                (d.as_secs(), d.subsec_nanos())
            }
            None => (UTIME_OMIT, 0),
        };
        let (mtime_sec, mtime_nsec) = match times.modified {
            Some(t) => {
                let d = t.as_duration();
                (d.as_secs(), d.subsec_nanos())
            }
            None => (UTIME_OMIT, 0),
        };
        Self { atime_sec, atime_nsec, _pad1: 0, mtime_sec, mtime_nsec, _pad2: 0 }
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

pub struct File(FileDesc);

#[derive(Clone)]
pub struct FileAttr {
    stat: AbiFileStat,
}

struct InnerReadDir {
    root: PathBuf,
    buf: Vec<u8>,
}

pub struct ReadDir {
    inner: Arc<InnerReadDir>,
    pos: usize,
}

pub struct DirEntry {
    root: PathBuf,
    name: OsString,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    accessed: Option<SystemTime>,
    modified: Option<SystemTime>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    mode: u32,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct FileType {
    mode: u32,
}

impl Hash for FileType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mode.hash(state);
    }
}

#[derive(Debug)]
pub struct DirBuilder {}

// ── FileAttr ──────────────────────────────────────────────────────────────────

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.stat.size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: self.stat.mode }
    }

    pub fn file_type(&self) -> FileType {
        FileType { mode: self.stat.mode & S_IFMT }
    }

    pub fn modified(&self) -> crate::io::Result<SystemTime> {
        Ok(SystemTime::from_timespec(self.stat.mtime.sec, self.stat.mtime.nsec))
    }

    pub fn accessed(&self) -> crate::io::Result<SystemTime> {
        Ok(SystemTime::from_timespec(self.stat.atime.sec, self.stat.atime.nsec))
    }

    pub fn created(&self) -> crate::io::Result<SystemTime> {
        Ok(SystemTime::from_timespec(self.stat.ctime.sec, self.stat.ctime.nsec))
    }
}

// ── FilePermissions ───────────────────────────────────────────────────────────

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.mode & 0o222 == 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.mode &= !0o222;
        } else {
            self.mode |= 0o200;
        }
    }
}

// ── FileTimes ─────────────────────────────────────────────────────────────────

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = Some(t);
    }
    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = Some(t);
    }
}

// ── FileType ──────────────────────────────────────────────────────────────────

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.mode == S_IFDIR
    }

    pub fn is_file(&self) -> bool {
        self.mode == S_IFREG
    }

    pub fn is_symlink(&self) -> bool {
        self.mode == S_IFLNK
    }
}

// ── OpenOptions ───────────────────────────────────────────────────────────────

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
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

    fn get_access_mode(&self) -> crate::io::Result<u32> {
        match (self.read, self.write, self.append) {
            (true, false, false) => Ok(O_RDONLY),
            (false, true, false) => Ok(O_WRONLY),
            (true, true, false) => Ok(O_RDWR),
            (false, _, true) => Ok(O_WRONLY | O_APPEND),
            (true, _, true) => Ok(O_RDWR | O_APPEND),
            (false, false, false) => {
                Err(io::const_error!(crate::io::ErrorKind::InvalidInput, "invalid access mode"))
            }
        }
    }

    fn get_creation_mode(&self) -> crate::io::Result<u32> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::const_error!(
                        crate::io::ErrorKind::InvalidInput,
                        "invalid creation mode"
                    ));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::const_error!(
                        crate::io::ErrorKind::InvalidInput,
                        "invalid creation mode"
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

// ── File ──────────────────────────────────────────────────────────────────────

impl File {
    pub fn as_raw_fd(&self) -> u32 {
        self.0.raw()
    }

    pub fn open(path: &Path, opts: &OpenOptions) -> crate::io::Result<File> {
        let path_str = path_to_str(path)?;
        let mut flags = opts.get_access_mode()?;
        flags |= opts.get_creation_mode()?;

        let ret = unsafe {
            raw_syscall6(
                SYS_FS_OPEN,
                path_str.as_ptr() as usize,
                path_str.len(),
                flags as usize,
                0,
                0,
                0,
            )
        };
        let fd = cvt(ret)? as u32;
        Ok(File(FileDesc::from_raw(fd)))
    }

    pub fn file_attr(&self) -> crate::io::Result<FileAttr> {
        let mut s = AbiFileStat::default();
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_STAT,
                self.0.raw() as usize,
                &mut s as *mut AbiFileStat as usize,
                0,
                0,
                0,
                0,
            )
        };
        cvt(ret)?;
        Ok(FileAttr { stat: s })
    }

    pub fn fsync(&self) -> crate::io::Result<()> {
        let ret =
            unsafe { raw_syscall6(SYS_FS_SYNC, self.0.raw() as usize, 0, 0, 0, 0, 0) };
        cvt(ret).map(|_| ())
    }

    pub fn datasync(&self) -> crate::io::Result<()> {
        self.fsync()
    }

    pub fn lock(&self) -> crate::io::Result<()> {
        let ret = unsafe {
            raw_syscall6(SYS_FS_FLOCK, self.0.raw() as usize, LOCK_EX as usize, 0, 0, 0, 0)
        };
        cvt(ret).map(|_| ())
    }

    pub fn lock_shared(&self) -> crate::io::Result<()> {
        let ret = unsafe {
            raw_syscall6(SYS_FS_FLOCK, self.0.raw() as usize, LOCK_SH as usize, 0, 0, 0, 0)
        };
        cvt(ret).map(|_| ())
    }

    pub fn try_lock(&self) -> Result<(), TryLockError> {
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_FLOCK,
                self.0.raw() as usize,
                (LOCK_EX | LOCK_NB) as usize,
                0,
                0,
                0,
                0,
            )
        };
        match cvt(ret) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == crate::io::ErrorKind::WouldBlock => {
                Err(TryLockError::WouldBlock)
            }
            Err(e) => Err(TryLockError::Error(e)),
        }
    }

    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_FLOCK,
                self.0.raw() as usize,
                (LOCK_SH | LOCK_NB) as usize,
                0,
                0,
                0,
                0,
            )
        };
        match cvt(ret) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == crate::io::ErrorKind::WouldBlock => {
                Err(TryLockError::WouldBlock)
            }
            Err(e) => Err(TryLockError::Error(e)),
        }
    }

    pub fn unlock(&self) -> crate::io::Result<()> {
        let ret = unsafe {
            raw_syscall6(SYS_FS_FLOCK, self.0.raw() as usize, LOCK_UN as usize, 0, 0, 0, 0)
        };
        cvt(ret).map(|_| ())
    }

    pub fn truncate(&self, size: u64) -> crate::io::Result<()> {
        let ret = unsafe {
            raw_syscall6(SYS_FS_FTRUNCATE, self.0.raw() as usize, size as usize, 0, 0, 0, 0)
        };
        cvt(ret).map(|_| ())
    }

    pub fn read(&self, buf: &mut [u8]) -> crate::io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_READ,
                self.0.raw() as usize,
                buf.as_mut_ptr() as usize,
                buf.len(),
                0,
                0,
                0,
            )
        };
        cvt(ret)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> crate::io::Result<usize> {
        // Build a kernel-compatible iovec array and dispatch SYS_FS_READV.
        let iovecs: vec::Vec<KernelIoVec> = bufs
            .iter()
            .map(|b| KernelIoVec { base: b.as_ptr() as usize, len: b.len() })
            .collect();
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_READV,
                self.0.raw() as usize,
                iovecs.as_ptr() as usize,
                iovecs.len(),
                0,
                0,
                0,
            )
        };
        cvt(ret)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn read_buf(&self, mut cursor: BorrowedCursor<'_>) -> crate::io::Result<()> {
        // SAFETY: the syscall initialises the bytes it writes.
        let buf = unsafe { cursor.as_mut() };
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_READ,
                self.0.raw() as usize,
                buf.as_mut_ptr() as usize,
                buf.len(),
                0,
                0,
                0,
            )
        };
        let n = cvt(ret)?;
        unsafe { cursor.advance_unchecked(n) };
        Ok(())
    }

    pub fn write(&self, buf: &[u8]) -> crate::io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_WRITE,
                self.0.raw() as usize,
                buf.as_ptr() as usize,
                buf.len(),
                0,
                0,
                0,
            )
        };
        cvt(ret)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> crate::io::Result<usize> {
        // Build a kernel-compatible iovec array and dispatch SYS_FS_WRITEV.
        let iovecs: vec::Vec<KernelIoVec> = bufs
            .iter()
            .map(|b| KernelIoVec { base: b.as_ptr() as usize, len: b.len() })
            .collect();
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_WRITEV,
                self.0.raw() as usize,
                iovecs.as_ptr() as usize,
                iovecs.len(),
                0,
                0,
                0,
            )
        };
        cvt(ret)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    pub fn flush(&self) -> crate::io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> crate::io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(off) => (SEEK_SET, off as i64),
            SeekFrom::End(off) => (SEEK_END, off),
            SeekFrom::Current(off) => (SEEK_CUR, off),
        };
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_SEEK,
                self.0.raw() as usize,
                offset as usize,
                whence as usize,
                0,
                0,
                0,
            )
        };
        cvt(ret).map(|v| v as u64)
    }

    pub fn size(&self) -> Option<crate::io::Result<u64>> {
        Some(self.file_attr().map(|a| a.size()))
    }

    pub fn tell(&self) -> crate::io::Result<u64> {
        self.seek(SeekFrom::Current(0))
    }

    pub fn duplicate(&self) -> crate::io::Result<File> {
        let ret =
            unsafe { raw_syscall6(SYS_FS_DUP, self.0.raw() as usize, 0, 0, 0, 0, 0) };
        let new_fd = cvt(ret)? as u32;
        Ok(File(FileDesc::from_raw(new_fd)))
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> crate::io::Result<()> {
        let ret = unsafe {
            raw_syscall6(SYS_FS_FCHMOD, self.0.raw() as usize, (perm.mode & 0o7777) as usize, 0, 0, 0, 0)
        };
        cvt(ret).map(|_| ())
    }

    pub fn set_times(&self, times: FileTimes) -> crate::io::Result<()> {
        let req = AbiUtimesRequest::from_file_times(&times);
        let ret = unsafe {
            raw_syscall6(SYS_FS_FUTIMES, self.0.raw() as usize, &req as *const AbiUtimesRequest as usize, 0, 0, 0, 0)
        };
        cvt(ret).map(|_| ())
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "File(fd={})", self.0.raw())
    }
}

// ── ReadDir ───────────────────────────────────────────────────────────────────

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReadDir({:?})", self.inner.root)
    }
}

impl Iterator for ReadDir {
    type Item = crate::io::Result<DirEntry>;

    fn next(&mut self) -> Option<crate::io::Result<DirEntry>> {
        loop {
            if self.pos >= self.inner.buf.len() {
                return None;
            }
            // Find the NUL terminator for this entry name.
            let start = self.pos;
            let mut end = start;
            while end < self.inner.buf.len() && self.inner.buf[end] != 0 {
                end += 1;
            }
            // Advance past the NUL byte.
            self.pos = end + 1;

            let name_bytes = &self.inner.buf[start..end];
            if name_bytes.is_empty() {
                continue; // skip empty / stray NUL bytes
            }
            let name_str = match core::str::from_utf8(name_bytes) {
                Ok(s) => s,
                Err(_) => continue, // skip malformed names
            };
            // Skip "." and ".." — callers rarely want them and can break loops.
            if name_str == "." || name_str == ".." {
                continue;
            }
            let name = OsString::from(name_str);
            return Some(Ok(DirEntry { root: self.inner.root.clone(), name }));
        }
    }
}

// ── DirEntry ──────────────────────────────────────────────────────────────────

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.root.join(&self.name)
    }

    pub fn file_name(&self) -> OsString {
        self.name.clone()
    }

    pub fn metadata(&self) -> crate::io::Result<FileAttr> {
        stat(&self.path())
    }

    pub fn file_type(&self) -> crate::io::Result<FileType> {
        self.metadata().map(|a| a.file_type())
    }
}

// ── DirBuilder ────────────────────────────────────────────────────────────────

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, path: &Path) -> crate::io::Result<()> {
        mkdir(path)
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Open a file descriptor for `path` (read-only), run `f`, then close it.
fn with_path_fd<T>(path: &Path, f: impl FnOnce(u32) -> crate::io::Result<T>) -> crate::io::Result<T> {
    let path_str = path_to_str(path)?;
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_OPEN,
            path_str.as_ptr() as usize,
            path_str.len(),
            O_RDONLY as usize,
            0,
            0,
            0,
        )
    };
    let fd = cvt(ret)? as u32;
    let _guard = FileDesc::from_raw(fd);
    f(fd)
}

pub fn stat(path: &Path) -> crate::io::Result<FileAttr> {
    with_path_fd(path, |fd| {
        let mut s = AbiFileStat::default();
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_STAT,
                fd as usize,
                &mut s as *mut AbiFileStat as usize,
                0,
                0,
                0,
                0,
            )
        };
        cvt(ret)?;
        Ok(FileAttr { stat: s })
    })
}

pub fn lstat(path: &Path) -> crate::io::Result<FileAttr> {
    let path_str = path_to_str(path)?;
    let mut s = AbiFileStat::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_LSTAT,
            path_str.as_ptr() as usize,
            path_str.len(),
            &mut s as *mut AbiFileStat as usize,
            0,
            0,
            0,
        )
    };
    cvt(ret)?;
    Ok(FileAttr { stat: s })
}

pub fn readdir(path: &Path) -> crate::io::Result<ReadDir> {
    let root = path.to_path_buf();
    let path_str = path_to_str(path)?;
    // Open the directory to get an fd.
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_OPEN,
            path_str.as_ptr() as usize,
            path_str.len(),
            O_RDONLY as usize,
            0,
            0,
            0,
        )
    };
    let fd = cvt(ret)? as u32;
    let desc = FileDesc::from_raw(fd);

    // Read directory entries in chunks until readdir returns 0 bytes.
    let chunk = 4096usize;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let old_len = buf.len();
        buf.resize(old_len + chunk, 0);
        let ret = unsafe {
            raw_syscall6(
                SYS_FS_READDIR,
                desc.raw() as usize,
                buf.as_mut_ptr().wrapping_add(old_len) as usize,
                chunk,
                0,
                0,
                0,
            )
        };
        match cvt(ret) {
            Ok(0) => {
                buf.truncate(old_len);
                break;
            }
            Ok(n) => {
                buf.truncate(old_len + n);
            }
            Err(e) => return Err(e),
        }
    }

    Ok(ReadDir { inner: Arc::new(InnerReadDir { root, buf }), pos: 0 })
}

pub fn mkdir(path: &Path) -> crate::io::Result<()> {
    let path_str = path_to_str(path)?;
    let ret = unsafe {
        raw_syscall6(SYS_FS_MKDIR, path_str.as_ptr() as usize, path_str.len(), 0, 0, 0, 0)
    };
    cvt(ret).map(|_| ())
}

pub fn unlink(path: &Path) -> crate::io::Result<()> {
    let path_str = path_to_str(path)?;
    let ret = unsafe {
        raw_syscall6(SYS_FS_UNLINK, path_str.as_ptr() as usize, path_str.len(), 0, 0, 0, 0)
    };
    cvt(ret).map(|_| ())
}

pub fn rmdir(path: &Path) -> crate::io::Result<()> {
    // On ThingOS, rmdir and unlink use the same VFS unlink syscall.
    unlink(path)
}

pub fn rename(old: &Path, new: &Path) -> crate::io::Result<()> {
    let old_str = path_to_str(old)?;
    let new_str = path_to_str(new)?;
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_RENAME,
            old_str.as_ptr() as usize,
            old_str.len(),
            new_str.as_ptr() as usize,
            new_str.len(),
            0,
            0,
        )
    };
    cvt(ret).map(|_| ())
}

pub fn canonicalize(path: &Path) -> crate::io::Result<PathBuf> {
    let path_str = path_to_str(path)?;
    let mut buf = vec![0u8; 4096];
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_REALPATH,
            path_str.as_ptr() as usize,
            path_str.len(),
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
        )
    };
    let len = cvt(ret)?;
    buf.truncate(len);
    // SAFETY: bytes came from kernel path API; valid OsString encoding.
    Ok(PathBuf::from(unsafe { OsString::from_encoded_bytes_unchecked(buf) }))
}

pub fn readlink(path: &Path) -> crate::io::Result<PathBuf> {
    let path_str = path_to_str(path)?;
    let mut buf = vec![0u8; 4096];
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_READLINK,
            path_str.as_ptr() as usize,
            path_str.len(),
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
        )
    };
    let n = cvt(ret)?;
    buf.truncate(n);
    Ok(PathBuf::from(unsafe { OsString::from_encoded_bytes_unchecked(buf) }))
}

pub fn symlink(src: &Path, dst: &Path) -> crate::io::Result<()> {
    let src_str = path_to_str(src)?;
    let dst_str = path_to_str(dst)?;
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_SYMLINK,
            src_str.as_ptr() as usize,
            src_str.len(),
            dst_str.as_ptr() as usize,
            dst_str.len(),
            0,
            0,
        )
    };
    cvt(ret).map(|_| ())
}

pub fn link(src: &Path, dst: &Path) -> crate::io::Result<()> {
    let src_str = path_to_str(src)?;
    let dst_str = path_to_str(dst)?;
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_LINK,
            src_str.as_ptr() as usize,
            src_str.len(),
            dst_str.as_ptr() as usize,
            dst_str.len(),
            0,
            0,
        )
    };
    cvt(ret).map(|_| ())
}

pub fn set_perm(path: &Path, perm: FilePermissions) -> crate::io::Result<()> {
    let path_str = path_to_str(path)?;
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_CHMOD,
            path_str.as_ptr() as usize,
            path_str.len(),
            perm.mode as usize,
            0,
            0,
            0,
        )
    };
    cvt(ret).map(|_| ())
}

pub fn set_times(path: &Path, times: FileTimes) -> crate::io::Result<()> {
    let path_str = path_to_str(path)?;
    let req = AbiUtimesRequest::from_file_times(&times);
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_UTIMES,
            path_str.as_ptr() as usize,
            path_str.len(),
            (&req as *const AbiUtimesRequest) as usize,
            0,
            0,
            0,
        )
    };
    cvt(ret).map(|_| ())
}

pub fn set_times_nofollow(path: &Path, times: FileTimes) -> crate::io::Result<()> {
    let path_str = path_to_str(path)?;
    let req = AbiUtimesRequest::from_file_times(&times);
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_LUTIMES,
            path_str.as_ptr() as usize,
            path_str.len(),
            (&req as *const AbiUtimesRequest) as usize,
            0,
            0,
            0,
        )
    };
    cvt(ret).map(|_| ())
}
