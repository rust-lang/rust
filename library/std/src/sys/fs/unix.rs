#![allow(nonstandard_style)]
#![allow(unsafe_op_in_unsafe_fn)]
#![cfg_attr(miri, allow(unused))]
#[cfg(test)]
mod tests;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use libc::c_char;
#[cfg(any(
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "android",
    target_os = "fuchsia",
    target_os = "hurd",
    target_os = "illumos",
    target_vendor = "apple",
))]
use libc::dirfd;
#[cfg(any(target_os = "fuchsia", target_os = "illumos", target_vendor = "apple"))]
use libc::fstatat as fstatat64;
#[cfg(any(all(target_os = "linux", not(target_env = "musl")), target_os = "hurd"))]
use libc::fstatat64;
#[cfg(any(
    target_os = "aix",
    target_os = "android",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "illumos",
    target_os = "nto",
    target_os = "redox",
    target_os = "solaris",
    target_os = "vita",
    all(target_os = "linux", target_env = "musl"),
))]
use libc::readdir as readdir64;
#[cfg(not(any(
    target_os = "aix",
    target_os = "android",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "hurd",
    target_os = "illumos",
    target_os = "l4re",
    target_os = "linux",
    target_os = "nto",
    target_os = "redox",
    target_os = "solaris",
    target_os = "vita",
)))]
use libc::readdir_r as readdir64_r;
#[cfg(any(all(target_os = "linux", not(target_env = "musl")), target_os = "hurd"))]
use libc::readdir64;
#[cfg(target_os = "l4re")]
use libc::readdir64_r;
use libc::{c_int, mode_t};
#[cfg(target_os = "android")]
use libc::{
    dirent as dirent64, fstat as fstat64, fstatat as fstatat64, ftruncate64, lseek64,
    lstat as lstat64, off64_t, open as open64, stat as stat64,
};
#[cfg(not(any(
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "l4re",
    target_os = "android",
    target_os = "hurd",
)))]
use libc::{
    dirent as dirent64, fstat as fstat64, ftruncate as ftruncate64, lseek as lseek64,
    lstat as lstat64, off_t as off64_t, open as open64, stat as stat64,
};
#[cfg(any(
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "l4re",
    target_os = "hurd"
))]
use libc::{dirent64, fstat64, ftruncate64, lseek64, lstat64, off64_t, open64, stat64};
use crate::ffi::{CStr, OsStr, OsString};
use crate::fmt::{self, Write as _};
use crate::fs::TryLockError;
use crate::io::{self, BorrowedCursor, Error, IoSlice, IoSliceMut, SeekFrom};
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd};
use crate::os::unix::prelude::*;
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::fd::FileDesc;
pub use crate::sys::fs::common::exists;
use crate::sys::time::SystemTime;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use crate::sys::weak::syscall;
#[cfg(target_os = "android")]
use crate::sys::weak::weak;
use crate::sys::{cvt, cvt_r};
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use crate::{mem, ptr};
pub struct File(FileDesc);
macro_rules! cfg_has_statx {
    ({ $($then_tt:tt)* } else { $($else_tt:tt)* }) => {
        cfg_select! {
            all(target_os = "linux", target_env = "gnu") => {
                $($then_tt)*
            }
            _ => {
                $($else_tt)*
            }
        }
    };
    ($($block_inner:tt)*) => {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            $($block_inner)*
        }
    };
}
cfg_has_statx! {{
    #[derive(Clone)]
    pub struct FileAttr {
        stat: stat64,
        statx_extra_fields: Option<StatxExtraFields>,
    }
    #[derive(Clone)]
    struct StatxExtraFields {
        stx_mask: u32,
        stx_btime: libc::statx_timestamp,
        #[cfg(target_pointer_width = "32")]
        stx_atime: libc::statx_timestamp,
        #[cfg(target_pointer_width = "32")]
        stx_ctime: libc::statx_timestamp,
        #[cfg(target_pointer_width = "32")]
        stx_mtime: libc::statx_timestamp,
    }
    unsafe fn try_statx(
        fd: c_int,
        path: *const c_char,
        flags: i32,
        mask: u32,
    ) -> Option<io::Result<FileAttr>> {
        use crate::sync::atomic::{Atomic, AtomicU8, Ordering};
        #[repr(u8)]
        enum STATX_STATE{ Unknown = 0, Present, Unavailable }
        static STATX_SAVED_STATE: Atomic<u8> = AtomicU8::new(STATX_STATE::Unknown as u8);
        syscall!(
            fn statx(
                fd: c_int,
                pathname: *const c_char,
                flags: c_int,
                mask: libc::c_uint,
                statxbuf: *mut libc::statx,
            ) -> c_int;
        );
        let statx_availability = STATX_SAVED_STATE.load(Ordering::Relaxed);
        if statx_availability == STATX_STATE::Unavailable as u8 {
            return None;
        }
        let mut buf: libc::statx = mem::zeroed();
        if let Err(err) = cvt(statx(fd, path, flags, mask, &mut buf)) {
            if STATX_SAVED_STATE.load(Ordering::Relaxed) == STATX_STATE::Present as u8 {
                return Some(Err(err));
            }
            let err2 = cvt(statx(0, ptr::null(), 0, libc::STATX_BASIC_STATS | libc::STATX_BTIME, ptr::null_mut()))
                .err()
                .and_then(|e| e.raw_os_error());
            if err2 == Some(libc::EFAULT) {
                STATX_SAVED_STATE.store(STATX_STATE::Present as u8, Ordering::Relaxed);
                return Some(Err(err));
            } else {
                STATX_SAVED_STATE.store(STATX_STATE::Unavailable as u8, Ordering::Relaxed);
                return None;
            }
        }
        if statx_availability == STATX_STATE::Unknown as u8 {
            STATX_SAVED_STATE.store(STATX_STATE::Present as u8, Ordering::Relaxed);
        }
        let mut stat: stat64 = mem::zeroed();
        stat.st_dev = libc::makedev(buf.stx_dev_major, buf.stx_dev_minor) as _;
        stat.st_ino = buf.stx_ino as libc::ino64_t;
        stat.st_nlink = buf.stx_nlink as libc::nlink_t;
        stat.st_mode = buf.stx_mode as libc::mode_t;
        stat.st_uid = buf.stx_uid as libc::uid_t;
        stat.st_gid = buf.stx_gid as libc::gid_t;
        stat.st_rdev = libc::makedev(buf.stx_rdev_major, buf.stx_rdev_minor) as _;
        stat.st_size = buf.stx_size as off64_t;
        stat.st_blksize = buf.stx_blksize as libc::blksize_t;
        stat.st_blocks = buf.stx_blocks as libc::blkcnt64_t;
        stat.st_atime = buf.stx_atime.tv_sec as libc::time_t;
        stat.st_atime_nsec = buf.stx_atime.tv_nsec as _;
        stat.st_mtime = buf.stx_mtime.tv_sec as libc::time_t;
        stat.st_mtime_nsec = buf.stx_mtime.tv_nsec as _;
        stat.st_ctime = buf.stx_ctime.tv_sec as libc::time_t;
        stat.st_ctime_nsec = buf.stx_ctime.tv_nsec as _;
        let extra = StatxExtraFields {
            stx_mask: buf.stx_mask,
            stx_btime: buf.stx_btime,
            #[cfg(target_pointer_width = "32")]
            stx_atime: buf.stx_atime,
            #[cfg(target_pointer_width = "32")]
            stx_ctime: buf.stx_ctime,
            #[cfg(target_pointer_width = "32")]
            stx_mtime: buf.stx_mtime,
        };
        Some(Ok(FileAttr { stat, statx_extra_fields: Some(extra) }))
    }
} else {
    #[derive(Clone)]
    pub struct FileAttr {
        stat: stat64,
    }
}}
struct InnerReadDir {
    dirp: Dir,
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
struct Dir(*mut libc::DIR);
unsafe impl Send for Dir {}
unsafe impl Sync for Dir {}
#[cfg(any(
    target_os = "aix",
    target_os = "android",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "hurd",
    target_os = "illumos",
    target_os = "linux",
    target_os = "nto",
    target_os = "redox",
    target_os = "solaris",
    target_os = "vita",
))]
pub struct DirEntry {
    dir: Arc<InnerReadDir>,
    entry: dirent64_min,
    name: crate::ffi::CString,
}
#[cfg(any(
    target_os = "aix",
    target_os = "android",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "hurd",
    target_os = "illumos",
    target_os = "linux",
    target_os = "nto",
    target_os = "redox",
    target_os = "solaris",
    target_os = "vita",
))]
struct dirent64_min {
    d_ino: u64,
    #[cfg(not(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "aix",
        target_os = "nto",
        target_os = "vita",
    )))]
    d_type: u8,
}
#[cfg(not(any(
    target_os = "aix",
    target_os = "android",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "hurd",
    target_os = "illumos",
    target_os = "linux",
    target_os = "nto",
    target_os = "redox",
    target_os = "solaris",
    target_os = "vita",
)))]
pub struct DirEntry {
    dir: Arc<InnerReadDir>,
    entry: dirent64,
}
#[derive(Clone)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    custom_flags: i32,
    mode: mode_t,
}
#[derive(Clone, PartialEq, Eq)]
pub struct FilePermissions {
    mode: mode_t,
}
#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    accessed: Option<SystemTime>,
    modified: Option<SystemTime>,
    #[cfg(target_vendor = "apple")]
    created: Option<SystemTime>,
}
#[derive(Copy, Clone, Eq)]
pub struct FileType {
    mode: mode_t,
}
impl PartialEq for FileType {
    fn eq(&self, other: &Self) -> bool {
        self.masked() == other.masked()
    }
}
impl core::hash::Hash for FileType {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.masked().hash(state);
    }
}
pub struct DirBuilder {
    mode: mode_t,
}
#[derive(Copy, Clone)]
struct Mode(mode_t);
cfg_has_statx! {{
    impl FileAttr {
        fn from_stat64(stat: stat64) -> Self {
            Self { stat, statx_extra_fields: None }
        }
        #[cfg(target_pointer_width = "32")]
        pub fn stx_mtime(&self) -> Option<&libc::statx_timestamp> {
            if let Some(ext) = &self.statx_extra_fields {
                if (ext.stx_mask & libc::STATX_MTIME) != 0 {
                    return Some(&ext.stx_mtime);
                }
            }
            None
        }
        #[cfg(target_pointer_width = "32")]
        pub fn stx_atime(&self) -> Option<&libc::statx_timestamp> {
            if let Some(ext) = &self.statx_extra_fields {
                if (ext.stx_mask & libc::STATX_ATIME) != 0 {
                    return Some(&ext.stx_atime);
                }
            }
            None
        }
        #[cfg(target_pointer_width = "32")]
        pub fn stx_ctime(&self) -> Option<&libc::statx_timestamp> {
            if let Some(ext) = &self.statx_extra_fields {
                if (ext.stx_mask & libc::STATX_CTIME) != 0 {
                    return Some(&ext.stx_ctime);
                }
            }
            None
        }
    }
} else {
    impl FileAttr {
        fn from_stat64(stat: stat64) -> Self {
            Self { stat }
        }
    }
}}
impl FileAttr {
    pub fn size(&self) -> u64 {
        self.stat.st_size as u64
    }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: (self.stat.st_mode as mode_t) }
    }
    pub fn file_type(&self) -> FileType {
        FileType { mode: self.stat.st_mode as mode_t }
    }
}
#[cfg(target_os = "netbsd")]
impl FileAttr {
    pub fn modified(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_mtime as i64, self.stat.st_mtimensec as i64)
    }
    pub fn accessed(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_atime as i64, self.stat.st_atimensec as i64)
    }
    pub fn created(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_birthtime as i64, self.stat.st_birthtimensec as i64)
    }
}
#[cfg(target_os = "aix")]
impl FileAttr {
    pub fn modified(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_mtime.tv_sec as i64, self.stat.st_mtime.tv_nsec as i64)
    }
    pub fn accessed(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_atime.tv_sec as i64, self.stat.st_atime.tv_nsec as i64)
    }
    pub fn created(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_ctime.tv_sec as i64, self.stat.st_ctime.tv_nsec as i64)
    }
}
#[cfg(not(any(target_os = "netbsd", target_os = "nto", target_os = "aix")))]
impl FileAttr {
    #[cfg(not(any(
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "vita",
        target_os = "hurd",
        target_os = "rtems",
        target_os = "nuttx",
    )))]
    pub fn modified(&self) -> io::Result<SystemTime> {
        #[cfg(target_pointer_width = "32")]
        cfg_has_statx! {
            if let Some(mtime) = self.stx_mtime() {
                return SystemTime::new(mtime.tv_sec, mtime.tv_nsec as i64);
            }
        }
        SystemTime::new(self.stat.st_mtime as i64, self.stat.st_mtime_nsec as i64)
    }
    #[cfg(any(
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "vita",
        target_os = "rtems",
    ))]
    pub fn modified(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_mtime as i64, 0)
    }
    #[cfg(any(target_os = "horizon", target_os = "hurd", target_os = "nuttx"))]
    pub fn modified(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_mtim.tv_sec as i64, self.stat.st_mtim.tv_nsec as i64)
    }
    #[cfg(not(any(
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "vita",
        target_os = "hurd",
        target_os = "rtems",
        target_os = "nuttx",
    )))]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        #[cfg(target_pointer_width = "32")]
        cfg_has_statx! {
            if let Some(atime) = self.stx_atime() {
                return SystemTime::new(atime.tv_sec, atime.tv_nsec as i64);
            }
        }
        SystemTime::new(self.stat.st_atime as i64, self.stat.st_atime_nsec as i64)
    }
    #[cfg(any(
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "vita",
        target_os = "rtems"
    ))]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_atime as i64, 0)
    }
    #[cfg(any(target_os = "horizon", target_os = "hurd", target_os = "nuttx"))]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_atim.tv_sec as i64, self.stat.st_atim.tv_nsec as i64)
    }
    #[cfg(any(
        target_os = "freebsd",
        target_os = "openbsd",
        target_vendor = "apple",
        target_os = "cygwin",
    ))]
    pub fn created(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_birthtime as i64, self.stat.st_birthtime_nsec as i64)
    }
    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "vita",
        target_vendor = "apple",
        target_os = "cygwin",
    )))]
    pub fn created(&self) -> io::Result<SystemTime> {
        cfg_has_statx! {
            if let Some(ext) = &self.statx_extra_fields {
                return if (ext.stx_mask & libc::STATX_BTIME) != 0 {
                    SystemTime::new(ext.stx_btime.tv_sec, ext.stx_btime.tv_nsec as i64)
                } else {
                    Err(io::const_error!(
                        io::ErrorKind::Unsupported,
                        "creation time is not available for the filesystem",
                    ))
                };
            }
        }
        Err(io::const_error!(
            io::ErrorKind::Unsupported,
            "creation time is not available on this platform currently",
        ))
    }
    #[cfg(target_os = "vita")]
    pub fn created(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_ctime as i64, 0)
    }
}
#[cfg(target_os = "nto")]
impl FileAttr {
    pub fn modified(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_mtim.tv_sec, self.stat.st_mtim.tv_nsec)
    }
    pub fn accessed(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_atim.tv_sec, self.stat.st_atim.tv_nsec)
    }
    pub fn created(&self) -> io::Result<SystemTime> {
        SystemTime::new(self.stat.st_ctim.tv_sec, self.stat.st_ctim.tv_nsec)
    }
}
impl AsInner<stat64> for FileAttr {
    #[inline]
    fn as_inner(&self) -> &stat64 {
        &self.stat
    }
}
impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.mode & 0o222 == 0
    }
    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.mode &= !0o222;
        } else {
            self.mode |= 0o222;
        }
    }
    pub fn mode(&self) -> u32 {
        self.mode as u32
    }
}
impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = Some(t);
    }
    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = Some(t);
    }
    #[cfg(target_vendor = "apple")]
    pub fn set_created(&mut self, t: SystemTime) {
        self.created = Some(t);
    }
}
impl FileType {
    pub fn is_dir(&self) -> bool {
        self.is(libc::S_IFDIR)
    }
    pub fn is_file(&self) -> bool {
        self.is(libc::S_IFREG)
    }
    pub fn is_symlink(&self) -> bool {
        self.is(libc::S_IFLNK)
    }
    pub fn is(&self, mode: mode_t) -> bool {
        self.masked() == mode
    }
    fn masked(&self) -> mode_t {
        self.mode & libc::S_IFMT
    }
}
impl fmt::Debug for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FileType { mode } = self;
        f.debug_struct("FileType").field("mode", &Mode(*mode)).finish()
    }
}
impl FromInner<u32> for FilePermissions {
    fn from_inner(mode: u32) -> FilePermissions {
        FilePermissions { mode: mode as mode_t }
    }
}
impl fmt::Debug for FilePermissions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FilePermissions { mode } = self;
        f.debug_struct("FilePermissions").field("mode", &Mode(*mode)).finish()
    }
}
impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.inner.root, f)
    }
}
impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;
    #[cfg(any(
        target_os = "aix",
        target_os = "android",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "hurd",
        target_os = "illumos",
        target_os = "linux",
        target_os = "nto",
        target_os = "redox",
        target_os = "solaris",
        target_os = "vita",
    ))]
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        use crate::sys::os::{errno, set_errno};
        if self.end_of_stream {
            return None;
        }
        unsafe {
            loop {
                set_errno(0);
                let entry_ptr: *const dirent64 = readdir64(self.inner.dirp.0);
                if entry_ptr.is_null() {
                    self.end_of_stream = true;
                    return match errno() {
                        0 => None,
                        e => Some(Err(Error::from_raw_os_error(e))),
                    };
                }
                let name = CStr::from_ptr((&raw const (*entry_ptr).d_name).cast());
                let name_bytes = name.to_bytes();
                if name_bytes == b"." || name_bytes == b".." {
                    continue;
                }
                #[cfg(not(target_os = "vita"))]
                let entry = dirent64_min {
                    #[cfg(target_os = "freebsd")]
                    d_ino: (*entry_ptr).d_fileno,
                    #[cfg(not(target_os = "freebsd"))]
                    d_ino: (*entry_ptr).d_ino as u64,
                    #[cfg(not(any(
                        target_os = "solaris",
                        target_os = "illumos",
                        target_os = "aix",
                        target_os = "nto",
                    )))]
                    d_type: (*entry_ptr).d_type as u8,
                };
                #[cfg(target_os = "vita")]
                let entry = dirent64_min { d_ino: 0u64 };
                return Some(Ok(DirEntry {
                    entry,
                    name: name.to_owned(),
                    dir: Arc::clone(&self.inner),
                }));
            }
        }
    }
    #[cfg(not(any(
        target_os = "aix",
        target_os = "android",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "hurd",
        target_os = "illumos",
        target_os = "linux",
        target_os = "nto",
        target_os = "redox",
        target_os = "solaris",
        target_os = "vita",
    )))]
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        if self.end_of_stream {
            return None;
        }
        unsafe {
            let mut ret = DirEntry { entry: mem::zeroed(), dir: Arc::clone(&self.inner) };
            let mut entry_ptr = ptr::null_mut();
            loop {
                let err = readdir64_r(self.inner.dirp.0, &mut ret.entry, &mut entry_ptr);
                if err != 0 {
                    if entry_ptr.is_null() {
                        self.end_of_stream = true;
                    }
                    return Some(Err(Error::from_raw_os_error(err)));
                }
                if entry_ptr.is_null() {
                    return None;
                }
                if ret.name_bytes() != b"." && ret.name_bytes() != b".." {
                    return Some(Ok(ret));
                }
            }
        }
    }
}
#[inline]
pub(crate) fn debug_assert_fd_is_open(fd: RawFd) {
    use crate::sys::os::errno;
    if core::ub_checks::check_library_ub() {
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } == -1 && errno() == libc::EBADF {
            rtabort!("IO Safety violation: owned file descriptor already closed");
        }
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        #[cfg(not(any(
            miri,
            target_os = "redox",
            target_os = "nto",
            target_os = "vita",
            target_os = "hurd",
            target_os = "espidf",
            target_os = "horizon",
            target_os = "vxworks",
            target_os = "rtems",
            target_os = "nuttx",
        )))]
        {
            let fd = unsafe { libc::dirfd(self.0) };
            debug_assert_fd_is_open(fd);
        }
        let r = unsafe { libc::closedir(self.0) };
        assert!(
            r == 0 || crate::io::Error::last_os_error().is_interrupted(),
            "unexpected error during closedir: {:?}",
            crate::io::Error::last_os_error()
        );
    }
}
impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.dir.root.join(self.file_name_os_str())
    }
    pub fn file_name(&self) -> OsString {
        self.file_name_os_str().to_os_string()
    }
    #[cfg(all(
        any(
            all(target_os = "linux", not(target_env = "musl")),
            target_os = "android",
            target_os = "fuchsia",
            target_os = "hurd",
            target_os = "illumos",
            target_vendor = "apple",
        ),
        not(miri)
    ))]
    pub fn metadata(&self) -> io::Result<FileAttr> {
        let fd = cvt(unsafe { dirfd(self.dir.dirp.0) })?;
        let name = self.name_cstr().as_ptr();
        cfg_has_statx! {
            if let Some(ret) = unsafe { try_statx(
                fd,
                name,
                libc::AT_SYMLINK_NOFOLLOW | libc::AT_STATX_SYNC_AS_STAT,
                libc::STATX_BASIC_STATS | libc::STATX_BTIME,
            ) } {
                return ret;
            }
        }
        let mut stat: stat64 = unsafe { mem::zeroed() };
        cvt(unsafe { fstatat64(fd, name, &mut stat, libc::AT_SYMLINK_NOFOLLOW) })?;
        Ok(FileAttr::from_stat64(stat))
    }
    #[cfg(any(
        not(any(
            all(target_os = "linux", not(target_env = "musl")),
            target_os = "android",
            target_os = "fuchsia",
            target_os = "hurd",
            target_os = "illumos",
            target_vendor = "apple",
        )),
        miri
    ))]
    pub fn metadata(&self) -> io::Result<FileAttr> {
        run_path_with_cstr(&self.path(), &lstat)
    }
    #[cfg(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks",
        target_os = "aix",
        target_os = "nto",
        target_os = "vita",
    ))]
    pub fn file_type(&self) -> io::Result<FileType> {
        self.metadata().map(|m| m.file_type())
    }
    #[cfg(not(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks",
        target_os = "aix",
        target_os = "nto",
        target_os = "vita",
    )))]
    pub fn file_type(&self) -> io::Result<FileType> {
        match self.entry.d_type {
            libc::DT_CHR => Ok(FileType { mode: libc::S_IFCHR }),
            libc::DT_FIFO => Ok(FileType { mode: libc::S_IFIFO }),
            libc::DT_LNK => Ok(FileType { mode: libc::S_IFLNK }),
            libc::DT_REG => Ok(FileType { mode: libc::S_IFREG }),
            libc::DT_SOCK => Ok(FileType { mode: libc::S_IFSOCK }),
            libc::DT_DIR => Ok(FileType { mode: libc::S_IFDIR }),
            libc::DT_BLK => Ok(FileType { mode: libc::S_IFBLK }),
            _ => self.metadata().map(|m| m.file_type()),
        }
    }
    #[cfg(any(
        target_os = "aix",
        target_os = "android",
        target_os = "cygwin",
        target_os = "emscripten",
        target_os = "espidf",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "haiku",
        target_os = "horizon",
        target_os = "hurd",
        target_os = "illumos",
        target_os = "l4re",
        target_os = "linux",
        target_os = "nto",
        target_os = "redox",
        target_os = "rtems",
        target_os = "solaris",
        target_os = "vita",
        target_os = "vxworks",
        target_vendor = "apple",
    ))]
    pub fn ino(&self) -> u64 {
        self.entry.d_ino as u64
    }
    #[cfg(any(target_os = "openbsd", target_os = "netbsd", target_os = "dragonfly"))]
    pub fn ino(&self) -> u64 {
        self.entry.d_fileno as u64
    }
    #[cfg(target_os = "nuttx")]
    pub fn ino(&self) -> u64 {
        0
    }
    #[cfg(any(
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_vendor = "apple",
    ))]
    fn name_bytes(&self) -> &[u8] {
        use crate::slice;
        unsafe {
            slice::from_raw_parts(
                self.entry.d_name.as_ptr() as *const u8,
                self.entry.d_namlen as usize,
            )
        }
    }
    #[cfg(not(any(
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_vendor = "apple",
    )))]
    fn name_bytes(&self) -> &[u8] {
        self.name_cstr().to_bytes()
    }
    #[cfg(not(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "fuchsia",
        target_os = "redox",
        target_os = "aix",
        target_os = "nto",
        target_os = "vita",
        target_os = "hurd",
    )))]
    fn name_cstr(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.entry.d_name.as_ptr()) }
    }
    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "fuchsia",
        target_os = "redox",
        target_os = "aix",
        target_os = "nto",
        target_os = "vita",
        target_os = "hurd",
    ))]
    fn name_cstr(&self) -> &CStr {
        &self.name
    }
    pub fn file_name_os_str(&self) -> &OsStr {
        OsStr::from_bytes(self.name_bytes())
    }
}
impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            custom_flags: 0,
            mode: 0o666,
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
    pub fn custom_flags(&mut self, flags: i32) {
        self.custom_flags = flags;
    }
    pub fn mode(&mut self, mode: u32) {
        self.mode = mode as mode_t;
    }
    fn get_access_mode(&self) -> io::Result<c_int> {
        match (self.read, self.write, self.append) {
            (true, false, false) => Ok(libc::O_RDONLY),
            (false, true, false) => Ok(libc::O_WRONLY),
            (true, true, false) => Ok(libc::O_RDWR),
            (false, _, true) => Ok(libc::O_WRONLY | libc::O_APPEND),
            (true, _, true) => Ok(libc::O_RDWR | libc::O_APPEND),
            (false, false, false) => {
                if self.create || self.create_new || self.truncate {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "creating or truncating a file requires write or append access",
                    ))
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "must specify at least one of read, write, or append access",
                    ))
                }
            }
        }
    }
    fn get_creation_mode(&self) -> io::Result<c_int> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "creating or truncating a file requires write or append access",
                    ));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "creating or truncating a file requires write or append access",
                    ));
                }
            }
        }
        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => 0,
            (true, false, false) => libc::O_CREAT,
            (false, true, false) => libc::O_TRUNC,
            (true, true, false) => libc::O_CREAT | libc::O_TRUNC,
            (_, _, true) => libc::O_CREAT | libc::O_EXCL,
        })
    }
}
impl fmt::Debug for OpenOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let OpenOptions { read, write, append, truncate, create, create_new, custom_flags, mode } =
            self;
        f.debug_struct("OpenOptions")
            .field("read", read)
            .field("write", write)
            .field("append", append)
            .field("truncate", truncate)
            .field("create", create)
            .field("create_new", create_new)
            .field("custom_flags", custom_flags)
            .field("mode", &Mode(*mode))
            .finish()
    }
}
impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        run_path_with_cstr(path, &|path| File::open_c(path, opts))
    }
    pub fn open_c(path: &CStr, opts: &OpenOptions) -> io::Result<File> {
        let flags = libc::O_CLOEXEC
            | opts.get_access_mode()?
            | opts.get_creation_mode()?
            | (opts.custom_flags as c_int & !libc::O_ACCMODE);
        let fd = cvt_r(|| unsafe { open64(path.as_ptr(), flags, opts.mode as c_int) })?;
        Ok(File(unsafe { FileDesc::from_raw_fd(fd) }))
    }
    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let fd = self.as_raw_fd();
        cfg_has_statx! {
            if let Some(ret) = unsafe { try_statx(
                fd,
                c"".as_ptr() as *const c_char,
                libc::AT_EMPTY_PATH | libc::AT_STATX_SYNC_AS_STAT,
                libc::STATX_BASIC_STATS | libc::STATX_BTIME,
            ) } {
                return ret;
            }
        }
        let mut stat: stat64 = unsafe { mem::zeroed() };
        cvt(unsafe { fstat64(fd, &mut stat) })?;
        Ok(FileAttr::from_stat64(stat))
    }
    pub fn fsync(&self) -> io::Result<()> {
        cvt_r(|| unsafe { os_fsync(self.as_raw_fd()) })?;
        return Ok(());
        #[cfg(target_vendor = "apple")]
        unsafe fn os_fsync(fd: c_int) -> c_int {
            libc::fcntl(fd, libc::F_FULLFSYNC)
        }
        #[cfg(not(target_vendor = "apple"))]
        unsafe fn os_fsync(fd: c_int) -> c_int {
            libc::fsync(fd)
        }
    }
    pub fn datasync(&self) -> io::Result<()> {
        cvt_r(|| unsafe { os_datasync(self.as_raw_fd()) })?;
        return Ok(());
        #[cfg(target_vendor = "apple")]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fcntl(fd, libc::F_FULLFSYNC)
        }
        #[cfg(any(
            target_os = "freebsd",
            target_os = "fuchsia",
            target_os = "linux",
            target_os = "cygwin",
            target_os = "android",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "nto",
            target_os = "hurd",
        ))]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fdatasync(fd)
        }
        #[cfg(not(any(
            target_os = "android",
            target_os = "fuchsia",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "cygwin",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "nto",
            target_os = "hurd",
            target_vendor = "apple",
        )))]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fsync(fd)
        }
    }
    #[cfg(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    ))]
    pub fn lock(&self) -> io::Result<()> {
        cvt(unsafe { libc::flock(self.as_raw_fd(), libc::LOCK_EX) })?;
        return Ok(());
    }
    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn lock(&self) -> io::Result<()> {
        let mut flock: libc::flock = unsafe { mem::zeroed() };
        flock.l_type = libc::F_WRLCK as libc::c_short;
        flock.l_whence = libc::SEEK_SET as libc::c_short;
        cvt(unsafe { libc::fcntl(self.as_raw_fd(), libc::F_SETLKW, &flock) })?;
        Ok(())
    }
    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    )))]
    pub fn lock(&self) -> io::Result<()> {
        Err(io::const_error!(io::ErrorKind::Unsupported, "lock() not supported"))
    }
    #[cfg(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    ))]
    pub fn lock_shared(&self) -> io::Result<()> {
        cvt(unsafe { libc::flock(self.as_raw_fd(), libc::LOCK_SH) })?;
        return Ok(());
    }
    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn lock_shared(&self) -> io::Result<()> {
        let mut flock: libc::flock = unsafe { mem::zeroed() };
        flock.l_type = libc::F_RDLCK as libc::c_short;
        flock.l_whence = libc::SEEK_SET as libc::c_short;
        cvt(unsafe { libc::fcntl(self.as_raw_fd(), libc::F_SETLKW, &flock) })?;
        Ok(())
    }
    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    )))]
    pub fn lock_shared(&self) -> io::Result<()> {
        Err(io::const_error!(io::ErrorKind::Unsupported, "lock_shared() not supported"))
    }
    #[cfg(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    ))]
    pub fn try_lock(&self) -> Result<(), TryLockError> {
        let result = cvt(unsafe { libc::flock(self.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) });
        if let Err(err) = result {
            if err.kind() == io::ErrorKind::WouldBlock {
                Err(TryLockError::WouldBlock)
            } else {
                Err(TryLockError::Error(err))
            }
        } else {
            Ok(())
        }
    }
    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn try_lock(&self) -> Result<(), TryLockError> {
        let mut flock: libc::flock = unsafe { mem::zeroed() };
        flock.l_type = libc::F_WRLCK as libc::c_short;
        flock.l_whence = libc::SEEK_SET as libc::c_short;
        let result = cvt(unsafe { libc::fcntl(self.as_raw_fd(), libc::F_SETLK, &flock) });
        if let Err(err) = result {
            if err.kind() == io::ErrorKind::WouldBlock {
                Err(TryLockError::WouldBlock)
            } else {
                Err(TryLockError::Error(err))
            }
        } else {
            Ok(())
        }
    }
    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    )))]
    pub fn try_lock(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(io::const_error!(
            io::ErrorKind::Unsupported,
            "try_lock() not supported"
        )))
    }
    #[cfg(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    ))]
    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        let result = cvt(unsafe { libc::flock(self.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) });
        if let Err(err) = result {
            if err.kind() == io::ErrorKind::WouldBlock {
                Err(TryLockError::WouldBlock)
            } else {
                Err(TryLockError::Error(err))
            }
        } else {
            Ok(())
        }
    }
    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        let mut flock: libc::flock = unsafe { mem::zeroed() };
        flock.l_type = libc::F_RDLCK as libc::c_short;
        flock.l_whence = libc::SEEK_SET as libc::c_short;
        let result = cvt(unsafe { libc::fcntl(self.as_raw_fd(), libc::F_SETLK, &flock) });
        if let Err(err) = result {
            if err.kind() == io::ErrorKind::WouldBlock {
                Err(TryLockError::WouldBlock)
            } else {
                Err(TryLockError::Error(err))
            }
        } else {
            Ok(())
        }
    }
    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    )))]
    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(io::const_error!(
            io::ErrorKind::Unsupported,
            "try_lock_shared() not supported"
        )))
    }
    #[cfg(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    ))]
    pub fn unlock(&self) -> io::Result<()> {
        cvt(unsafe { libc::flock(self.as_raw_fd(), libc::LOCK_UN) })?;
        return Ok(());
    }
    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn unlock(&self) -> io::Result<()> {
        let mut flock: libc::flock = unsafe { mem::zeroed() };
        flock.l_type = libc::F_UNLCK as libc::c_short;
        flock.l_whence = libc::SEEK_SET as libc::c_short;
        cvt(unsafe { libc::fcntl(self.as_raw_fd(), libc::F_SETLKW, &flock) })?;
        Ok(())
    }
    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "cygwin",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "aix",
        target_vendor = "apple",
    )))]
    pub fn unlock(&self) -> io::Result<()> {
        Err(io::const_error!(io::ErrorKind::Unsupported, "unlock() not supported"))
    }
    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let size: off64_t =
            size.try_into().map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        cvt_r(|| unsafe { ftruncate64(self.as_raw_fd(), size) }).map(drop)
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
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.0.read_at(buf, offset)
    }
    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.0.read_buf(cursor)
    }
    pub fn read_buf_at(&self, cursor: BorrowedCursor<'_>, offset: u64) -> io::Result<()> {
        self.0.read_buf_at(cursor, offset)
    }
    pub fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        self.0.read_vectored_at(bufs, offset)
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
    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.0.write_at(buf, offset)
    }
    pub fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        self.0.write_vectored_at(bufs, offset)
    }
    #[inline]
    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            SeekFrom::Start(off) => (libc::SEEK_SET, off as i64),
            SeekFrom::End(off) => (libc::SEEK_END, off),
            SeekFrom::Current(off) => (libc::SEEK_CUR, off),
        };
        let n = cvt(unsafe { lseek64(self.as_raw_fd(), pos as off64_t, whence) })?;
        Ok(n as u64)
    }
    pub fn size(&self) -> Option<io::Result<u64>> {
        match self.file_attr().map(|attr| attr.size()) {
            Ok(0) => None,
            result => Some(result),
        }
    }
    pub fn tell(&self) -> io::Result<u64> {
        self.seek(SeekFrom::Current(0))
    }
    pub fn duplicate(&self) -> io::Result<File> {
        self.0.duplicate().map(File)
    }
    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        cvt_r(|| unsafe { libc::fchmod(self.as_raw_fd(), perm.mode) })?;
        Ok(())
    }
    pub fn set_times(&self, times: FileTimes) -> io::Result<()> {
        cfg_select! {
            any(target_os = "redox", target_os = "espidf", target_os = "horizon", target_os = "nuttx") => {
                let _ = times;
                Err(io::const_error!(
                    io::ErrorKind::Unsupported,
                    "setting file times not supported",
                ))
            }
            target_vendor = "apple" => {
                let ta = TimesAttrlist::from_times(&times)?;
                cvt(unsafe { libc::fsetattrlist(
                    self.as_raw_fd(),
                    ta.attrlist(),
                    ta.times_buf(),
                    ta.times_buf_size(),
                    0
                ) })?;
                Ok(())
            }
            target_os = "android" => {
                let times = [file_time_to_timespec(times.accessed)?, file_time_to_timespec(times.modified)?];
                cvt(unsafe {
                    weak!(
                        fn futimens(fd: c_int, times: *const libc::timespec) -> c_int;
                    );
                    match futimens.get() {
                        Some(futimens) => futimens(self.as_raw_fd(), times.as_ptr()),
                        None => return Err(io::const_error!(
                            io::ErrorKind::Unsupported,
                            "setting file times requires Android API level >= 19",
                        )),
                    }
                })?;
                Ok(())
            }
            _ => {
                #[cfg(all(target_os = "linux", target_env = "gnu", target_pointer_width = "32", not(target_arch = "riscv32")))]
                {
                    use crate::sys::{time::__timespec64, weak::weak};
                    weak!(
                        fn __futimens64(fd: c_int, times: *const __timespec64) -> c_int;
                    );
                    if let Some(futimens64) = __futimens64.get() {
                        let to_timespec = |time: Option<SystemTime>| time.map(|time| time.t.to_timespec64())
                            .unwrap_or(__timespec64::new(0, libc::UTIME_OMIT as _));
                        let times = [to_timespec(times.accessed), to_timespec(times.modified)];
                        cvt(unsafe { futimens64(self.as_raw_fd(), times.as_ptr()) })?;
                        return Ok(());
                    }
                }
                let times = [file_time_to_timespec(times.accessed)?, file_time_to_timespec(times.modified)?];
                cvt(unsafe { libc::futimens(self.as_raw_fd(), times.as_ptr()) })?;
                Ok(())
            }
        }
    }
}
#[cfg(not(any(
    target_os = "redox",
    target_os = "espidf",
    target_os = "horizon",
    target_os = "nuttx",
)))]
fn file_time_to_timespec(time: Option<SystemTime>) -> io::Result<libc::timespec> {
    match time {
        Some(time) if let Some(ts) = time.t.to_timespec() => Ok(ts),
        Some(time) if time > crate::sys::time::UNIX_EPOCH => Err(io::const_error!(
            io::ErrorKind::InvalidInput,
            "timestamp is too large to set as a file time",
        )),
        Some(_) => Err(io::const_error!(
            io::ErrorKind::InvalidInput,
            "timestamp is too small to set as a file time",
        )),
        None => Ok(libc::timespec { tv_sec: 0, tv_nsec: libc::UTIME_OMIT as _ }),
    }
}
#[cfg(target_vendor = "apple")]
struct TimesAttrlist {
    buf: [mem::MaybeUninit<libc::timespec>; 3],
    attrlist: libc::attrlist,
    num_times: usize,
}
#[cfg(target_vendor = "apple")]
impl TimesAttrlist {
    fn from_times(times: &FileTimes) -> io::Result<Self> {
        let mut this = Self {
            buf: [mem::MaybeUninit::<libc::timespec>::uninit(); 3],
            attrlist: unsafe { mem::zeroed() },
            num_times: 0,
        };
        this.attrlist.bitmapcount = libc::ATTR_BIT_MAP_COUNT;
        if times.created.is_some() {
            this.buf[this.num_times].write(file_time_to_timespec(times.created)?);
            this.num_times += 1;
            this.attrlist.commonattr |= libc::ATTR_CMN_CRTIME;
        }
        if times.modified.is_some() {
            this.buf[this.num_times].write(file_time_to_timespec(times.modified)?);
            this.num_times += 1;
            this.attrlist.commonattr |= libc::ATTR_CMN_MODTIME;
        }
        if times.accessed.is_some() {
            this.buf[this.num_times].write(file_time_to_timespec(times.accessed)?);
            this.num_times += 1;
            this.attrlist.commonattr |= libc::ATTR_CMN_ACCTIME;
        }
        Ok(this)
    }
    fn attrlist(&self) -> *mut libc::c_void {
        (&raw const self.attrlist).cast::<libc::c_void>().cast_mut()
    }
    fn times_buf(&self) -> *mut libc::c_void {
        self.buf.as_ptr().cast::<libc::c_void>().cast_mut()
    }
    fn times_buf_size(&self) -> usize {
        self.num_times * size_of::<libc::timespec>()
    }
}
impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder { mode: 0o777 }
    }
    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        run_path_with_cstr(p, &|p| cvt(unsafe { libc::mkdir(p.as_ptr(), self.mode) }).map(|_| ()))
    }
    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode as mode_t;
    }
}
impl fmt::Debug for DirBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let DirBuilder { mode } = self;
        f.debug_struct("DirBuilder").field("mode", &Mode(*mode)).finish()
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
        Self(FromRawFd::from_raw_fd(raw_fd))
    }
}
impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(any(target_os = "linux", target_os = "illumos", target_os = "solaris"))]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            let mut p = PathBuf::from("/proc/self/fd");
            p.push(&fd.to_string());
            run_path_with_cstr(&p, &readlink).ok()
        }
        #[cfg(any(target_vendor = "apple", target_os = "netbsd"))]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            let mut buf = vec![0; libc::PATH_MAX as usize];
            let n = unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_ptr()) };
            if n == -1 {
                cfg_select! {
                    target_os = "netbsd" => {
                        let mut p = PathBuf::from("/proc/self/fd");
                        p.push(&fd.to_string());
                        return run_path_with_cstr(&p, &readlink).ok()
                    }
                    _ => {
                        return None;
                    }
                }
            }
            let l = buf.iter().position(|&c| c == 0).unwrap();
            buf.truncate(l as usize);
            buf.shrink_to_fit();
            Some(PathBuf::from(OsString::from_vec(buf)))
        }
        #[cfg(target_os = "freebsd")]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            let info = Box::<libc::kinfo_file>::new_zeroed();
            let mut info = unsafe { info.assume_init() };
            info.kf_structsize = size_of::<libc::kinfo_file>() as libc::c_int;
            let n = unsafe { libc::fcntl(fd, libc::F_KINFO, &mut *info) };
            if n == -1 {
                return None;
            }
            let buf = unsafe { CStr::from_ptr(info.kf_path.as_mut_ptr()).to_bytes().to_vec() };
            Some(PathBuf::from(OsString::from_vec(buf)))
        }
        #[cfg(target_os = "vxworks")]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            let mut buf = vec![0; libc::PATH_MAX as usize];
            let n = unsafe { libc::ioctl(fd, libc::FIOGETNAME, buf.as_ptr()) };
            if n == -1 {
                return None;
            }
            let l = buf.iter().position(|&c| c == 0).unwrap();
            buf.truncate(l as usize);
            Some(PathBuf::from(OsString::from_vec(buf)))
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "vxworks",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "illumos",
            target_os = "solaris",
            target_vendor = "apple",
        )))]
        fn get_path(_fd: c_int) -> Option<PathBuf> {
            None
        }
        fn get_mode(fd: c_int) -> Option<(bool, bool)> {
            let mode = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if mode == -1 {
                return None;
            }
            match mode & libc::O_ACCMODE {
                libc::O_RDONLY => Some((true, false)),
                libc::O_RDWR => Some((true, true)),
                libc::O_WRONLY => Some((false, true)),
                _ => None,
            }
        }
        let fd = self.as_raw_fd();
        let mut b = f.debug_struct("File");
        b.field("fd", &fd);
        if let Some(path) = get_path(fd) {
            b.field("path", &path);
        }
        if let Some((read, write)) = get_mode(fd) {
            b.field("read", &read).field("write", &write);
        }
        b.finish()
    }
}
impl fmt::Debug for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(mode) = *self;
        write!(f, "0o{mode:06o}")?;
        let entry_type = match mode & libc::S_IFMT {
            libc::S_IFDIR => 'd',
            libc::S_IFBLK => 'b',
            libc::S_IFCHR => 'c',
            libc::S_IFLNK => 'l',
            libc::S_IFIFO => 'p',
            libc::S_IFREG => '-',
            _ => return Ok(()),
        };
        f.write_str(" (")?;
        f.write_char(entry_type)?;
        f.write_char(if mode & libc::S_IRUSR != 0 { 'r' } else { '-' })?;
        f.write_char(if mode & libc::S_IWUSR != 0 { 'w' } else { '-' })?;
        let owner_executable = mode & libc::S_IXUSR != 0;
        let setuid = mode as c_int & libc::S_ISUID as c_int != 0;
        f.write_char(match (owner_executable, setuid) {
            (true, true) => 's',
            (false, true) => 'S',
            (true, false) => 'x',
            (false, false) => '-',
        })?;
        f.write_char(if mode & libc::S_IRGRP != 0 { 'r' } else { '-' })?;
        f.write_char(if mode & libc::S_IWGRP != 0 { 'w' } else { '-' })?;
        let group_executable = mode & libc::S_IXGRP != 0;
        let setgid = mode as c_int & libc::S_ISGID as c_int != 0;
        f.write_char(match (group_executable, setgid) {
            (true, true) => 's',
            (false, true) => 'S',
            (true, false) => 'x',
            (false, false) => '-',
        })?;
        f.write_char(if mode & libc::S_IROTH != 0 { 'r' } else { '-' })?;
        f.write_char(if mode & libc::S_IWOTH != 0 { 'w' } else { '-' })?;
        let other_executable = mode & libc::S_IXOTH != 0;
        let sticky = mode as c_int & libc::S_ISVTX as c_int != 0;
        f.write_char(match (entry_type, other_executable, sticky) {
            ('d', true, true) => 't',
            ('d', false, true) => 'T',
            (_, true, _) => 'x',
            (_, false, _) => '-',
        })?;
        f.write_char(')')
    }
}
pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    let ptr = run_path_with_cstr(path, &|p| unsafe { Ok(libc::opendir(p.as_ptr())) })?;
    if ptr.is_null() {
        Err(Error::last_os_error())
    } else {
        let root = path.to_path_buf();
        let inner = InnerReadDir { dirp: Dir(ptr), root };
        Ok(ReadDir::new(inner))
    }
}
pub fn unlink(p: &CStr) -> io::Result<()> {
    cvt(unsafe { libc::unlink(p.as_ptr()) }).map(|_| ())
}
pub fn rename(old: &CStr, new: &CStr) -> io::Result<()> {
    cvt(unsafe { libc::rename(old.as_ptr(), new.as_ptr()) }).map(|_| ())
}
pub fn set_perm(p: &CStr, perm: FilePermissions) -> io::Result<()> {
    cvt_r(|| unsafe { libc::chmod(p.as_ptr(), perm.mode) }).map(|_| ())
}
pub fn rmdir(p: &CStr) -> io::Result<()> {
    cvt(unsafe { libc::rmdir(p.as_ptr()) }).map(|_| ())
}
pub fn readlink(c_path: &CStr) -> io::Result<PathBuf> {
    let p = c_path.as_ptr();
    let mut buf = Vec::with_capacity(256);
    loop {
        let buf_read =
            cvt(unsafe { libc::readlink(p, buf.as_mut_ptr() as *mut _, buf.capacity()) })? as usize;
        unsafe {
            buf.set_len(buf_read);
        }
        if buf_read != buf.capacity() {
            buf.shrink_to_fit();
            return Ok(PathBuf::from(OsString::from_vec(buf)));
        }
        buf.reserve(1);
    }
}
pub fn symlink(original: &CStr, link: &CStr) -> io::Result<()> {
    cvt(unsafe { libc::symlink(original.as_ptr(), link.as_ptr()) }).map(|_| ())
}
pub fn link(original: &CStr, link: &CStr) -> io::Result<()> {
    cfg_select! {
        any(target_os = "vxworks", target_os = "redox", target_os = "android", target_os = "espidf", target_os = "horizon", target_os = "vita", target_env = "nto70") => {
            cvt(unsafe { libc::link(original.as_ptr(), link.as_ptr()) })?;
        }
        _ => {
            cvt(unsafe { libc::linkat(libc::AT_FDCWD, original.as_ptr(), libc::AT_FDCWD, link.as_ptr(), 0) })?;
        }
    }
    Ok(())
}
pub fn stat(p: &CStr) -> io::Result<FileAttr> {
    cfg_has_statx! {
        if let Some(ret) = unsafe { try_statx(
            libc::AT_FDCWD,
            p.as_ptr(),
            libc::AT_STATX_SYNC_AS_STAT,
            libc::STATX_BASIC_STATS | libc::STATX_BTIME,
        ) } {
            return ret;
        }
    }
    let mut stat: stat64 = unsafe { mem::zeroed() };
    cvt(unsafe { stat64(p.as_ptr(), &mut stat) })?;
    Ok(FileAttr::from_stat64(stat))
}
pub fn lstat(p: &CStr) -> io::Result<FileAttr> {
    cfg_has_statx! {
        if let Some(ret) = unsafe { try_statx(
            libc::AT_FDCWD,
            p.as_ptr(),
            libc::AT_SYMLINK_NOFOLLOW | libc::AT_STATX_SYNC_AS_STAT,
            libc::STATX_BASIC_STATS | libc::STATX_BTIME,
        ) } {
            return ret;
        }
    }
    let mut stat: stat64 = unsafe { mem::zeroed() };
    cvt(unsafe { lstat64(p.as_ptr(), &mut stat) })?;
    Ok(FileAttr::from_stat64(stat))
}
pub fn canonicalize(path: &CStr) -> io::Result<PathBuf> {
    let r = unsafe { libc::realpath(path.as_ptr(), ptr::null_mut()) };
    if r.is_null() {
        return Err(io::Error::last_os_error());
    }
    Ok(PathBuf::from(OsString::from_vec(unsafe {
        let buf = CStr::from_ptr(r).to_bytes().to_vec();
        libc::free(r as *mut _);
        buf
    })))
}
fn open_from(from: &Path) -> io::Result<(crate::fs::File, crate::fs::Metadata)> {
    use crate::fs::File;
    use crate::sys::fs::common::NOT_FILE_ERROR;
    let reader = File::open(from)?;
    let metadata = reader.metadata()?;
    if !metadata.is_file() {
        return Err(NOT_FILE_ERROR);
    }
    Ok((reader, metadata))
}
fn set_times_impl(p: &CStr, times: FileTimes, follow_symlinks: bool) -> io::Result<()> {
    cfg_select! {
       any(target_os = "redox", target_os = "espidf", target_os = "horizon", target_os = "nuttx") => {
            let _ = (p, times, follow_symlinks);
            Err(io::const_error!(
                io::ErrorKind::Unsupported,
                "setting file times not supported",
            ))
       }
       target_vendor = "apple" => {
            let ta = TimesAttrlist::from_times(&times)?;
            let options = if follow_symlinks {
                0
            } else {
                libc::FSOPT_NOFOLLOW
            };
            cvt(unsafe { libc::setattrlist(
                p.as_ptr(),
                ta.attrlist(),
                ta.times_buf(),
                ta.times_buf_size(),
                options as u32
            ) })?;
            Ok(())
       }
       target_os = "android" => {
            let times = [file_time_to_timespec(times.accessed)?, file_time_to_timespec(times.modified)?];
            let flags = if follow_symlinks { 0 } else { libc::AT_SYMLINK_NOFOLLOW };
            cvt(unsafe {
                weak!(
                    fn utimensat(dirfd: c_int, path: *const libc::c_char, times: *const libc::timespec, flags: c_int) -> c_int;
                );
                match utimensat.get() {
                    Some(utimensat) => utimensat(libc::AT_FDCWD, p.as_ptr(), times.as_ptr(), flags),
                    None => return Err(io::const_error!(
                        io::ErrorKind::Unsupported,
                        "setting file times requires Android API level >= 19",
                    )),
                }
            })?;
            Ok(())
       }
       _ => {
            let flags = if follow_symlinks { 0 } else { libc::AT_SYMLINK_NOFOLLOW };
            #[cfg(all(target_os = "linux", target_env = "gnu", target_pointer_width = "32", not(target_arch = "riscv32")))]
            {
                use crate::sys::{time::__timespec64, weak::weak};
                weak!(
                    fn __utimensat64(dirfd: c_int, path: *const c_char, times: *const __timespec64, flags: c_int) -> c_int;
                );
                if let Some(utimensat64) = __utimensat64.get() {
                    let to_timespec = |time: Option<SystemTime>| time.map(|time| time.t.to_timespec64())
                        .unwrap_or(__timespec64::new(0, libc::UTIME_OMIT as _));
                    let times = [to_timespec(times.accessed), to_timespec(times.modified)];
                    cvt(unsafe { utimensat64(libc::AT_FDCWD, p.as_ptr(), times.as_ptr(), flags) })?;
                    return Ok(());
                }
            }
            let times = [file_time_to_timespec(times.accessed)?, file_time_to_timespec(times.modified)?];
            cvt(unsafe { libc::utimensat(libc::AT_FDCWD, p.as_ptr(), times.as_ptr(), flags) })?;
            Ok(())
         }
    }
}
#[inline(always)]
pub fn set_times(p: &CStr, times: FileTimes) -> io::Result<()> {
    set_times_impl(p, times, true)
}
#[inline(always)]
pub fn set_times_nofollow(p: &CStr, times: FileTimes) -> io::Result<()> {
    set_times_impl(p, times, false)
}
#[cfg(target_os = "espidf")]
fn open_to_and_set_permissions(
    to: &Path,
    _reader_metadata: &crate::fs::Metadata,
) -> io::Result<(crate::fs::File, crate::fs::Metadata)> {
    use crate::fs::OpenOptions;
    let writer = OpenOptions::new().open(to)?;
    let writer_metadata = writer.metadata()?;
    Ok((writer, writer_metadata))
}
#[cfg(not(target_os = "espidf"))]
fn open_to_and_set_permissions(
    to: &Path,
    reader_metadata: &crate::fs::Metadata,
) -> io::Result<(crate::fs::File, crate::fs::Metadata)> {
    use crate::fs::OpenOptions;
    use crate::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let perm = reader_metadata.permissions();
    let writer = OpenOptions::new()
        .mode(perm.mode())
        .write(true)
        .create(true)
        .truncate(true)
        .open(to)?;
    let writer_metadata = writer.metadata()?;
    #[cfg(not(target_os = "vita"))]
    if writer_metadata.is_file() {
        writer.set_permissions(perm)?;
    }
    Ok((writer, writer_metadata))
}
mod cfm {
    use crate::fs::{File, Metadata};
    use crate::io::{BorrowedCursor, IoSlice, IoSliceMut, Read, Result, Write};
    #[allow(dead_code)]
    pub struct CachedFileMetadata(pub File, pub Metadata);
    impl Read for CachedFileMetadata {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            self.0.read(buf)
        }
        fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> Result<usize> {
            self.0.read_vectored(bufs)
        }
        fn read_buf(&mut self, cursor: BorrowedCursor<'_>) -> Result<()> {
            self.0.read_buf(cursor)
        }
        #[inline]
        fn is_read_vectored(&self) -> bool {
            self.0.is_read_vectored()
        }
        fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize> {
            self.0.read_to_end(buf)
        }
        fn read_to_string(&mut self, buf: &mut String) -> Result<usize> {
            self.0.read_to_string(buf)
        }
    }
    impl Write for CachedFileMetadata {
        fn write(&mut self, buf: &[u8]) -> Result<usize> {
            self.0.write(buf)
        }
        fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> Result<usize> {
            self.0.write_vectored(bufs)
        }
        #[inline]
        fn is_write_vectored(&self) -> bool {
            self.0.is_write_vectored()
        }
        #[inline]
        fn flush(&mut self) -> Result<()> {
            self.0.flush()
        }
    }
}
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(in crate::sys) use cfm::CachedFileMetadata;
#[cfg(not(target_vendor = "apple"))]
pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    let (reader, reader_metadata) = open_from(from)?;
    let (writer, writer_metadata) = open_to_and_set_permissions(to, &reader_metadata)?;
    io::copy(
        &mut cfm::CachedFileMetadata(reader, reader_metadata),
        &mut cfm::CachedFileMetadata(writer, writer_metadata),
    )
}
#[cfg(target_vendor = "apple")]
pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    const COPYFILE_ALL: libc::copyfile_flags_t = libc::COPYFILE_METADATA | libc::COPYFILE_DATA;
    struct FreeOnDrop(libc::copyfile_state_t);
    impl Drop for FreeOnDrop {
        fn drop(&mut self) {
            unsafe {
                libc::copyfile_state_free(self.0);
            }
        }
    }
    let (reader, reader_metadata) = open_from(from)?;
    let clonefile_result = run_path_with_cstr(to, &|to| {
        cvt(unsafe { libc::fclonefileat(reader.as_raw_fd(), libc::AT_FDCWD, to.as_ptr(), 0) })
    });
    match clonefile_result {
        Ok(_) => return Ok(reader_metadata.len()),
        Err(e) => match e.raw_os_error() {
            Some(libc::ENOTSUP) | Some(libc::EEXIST) | Some(libc::EXDEV) => (),
            _ => return Err(e),
        },
    }
    let (writer, writer_metadata) = open_to_and_set_permissions(to, &reader_metadata)?;
    let state = unsafe {
        let state = libc::copyfile_state_alloc();
        if state.is_null() {
            return Err(crate::io::Error::last_os_error());
        }
        FreeOnDrop(state)
    };
    let flags = if writer_metadata.is_file() { COPYFILE_ALL } else { libc::COPYFILE_DATA };
    cvt(unsafe { libc::fcopyfile(reader.as_raw_fd(), writer.as_raw_fd(), state.0, flags) })?;
    let mut bytes_copied: libc::off_t = 0;
    cvt(unsafe {
        libc::copyfile_state_get(
            state.0,
            libc::COPYFILE_STATE_COPIED as u32,
            (&raw mut bytes_copied) as *mut libc::c_void,
        )
    })?;
    Ok(bytes_copied as u64)
}
pub fn chown(path: &Path, uid: u32, gid: u32) -> io::Result<()> {
    run_path_with_cstr(path, &|path| {
        cvt(unsafe { libc::chown(path.as_ptr(), uid as libc::uid_t, gid as libc::gid_t) })
            .map(|_| ())
    })
}
pub fn fchown(fd: c_int, uid: u32, gid: u32) -> io::Result<()> {
    cvt(unsafe { libc::fchown(fd, uid as libc::uid_t, gid as libc::gid_t) })?;
    Ok(())
}
#[cfg(not(target_os = "vxworks"))]
pub fn lchown(path: &Path, uid: u32, gid: u32) -> io::Result<()> {
    run_path_with_cstr(path, &|path| {
        cvt(unsafe { libc::lchown(path.as_ptr(), uid as libc::uid_t, gid as libc::gid_t) })
            .map(|_| ())
    })
}
#[cfg(target_os = "vxworks")]
pub fn lchown(path: &Path, uid: u32, gid: u32) -> io::Result<()> {
    Err(io::const_error!(io::ErrorKind::Unsupported, "lchown not supported by vxworks"))
}
#[cfg(not(any(target_os = "fuchsia", target_os = "vxworks")))]
pub fn chroot(dir: &Path) -> io::Result<()> {
    run_path_with_cstr(dir, &|dir| cvt(unsafe { libc::chroot(dir.as_ptr()) }).map(|_| ()))
}
#[cfg(target_os = "vxworks")]
pub fn chroot(dir: &Path) -> io::Result<()> {
    Err(io::const_error!(io::ErrorKind::Unsupported, "chroot not supported by vxworks"))
}
pub fn mkfifo(path: &Path, mode: u32) -> io::Result<()> {
    run_path_with_cstr(path, &|path| {
        cvt(unsafe { libc::mkfifo(path.as_ptr(), mode.try_into().unwrap()) }).map(|_| ())
    })
}
pub use remove_dir_impl::remove_dir_all;
#[cfg(any(
    target_os = "redox",
    target_os = "espidf",
    target_os = "horizon",
    target_os = "vita",
    target_os = "nto",
    target_os = "vxworks",
    miri
))]
mod remove_dir_impl {
    pub use crate::sys::fs::common::remove_dir_all;
}
#[cfg(not(any(
    target_os = "redox",
    target_os = "espidf",
    target_os = "horizon",
    target_os = "vita",
    target_os = "nto",
    target_os = "vxworks",
    miri
)))]
mod remove_dir_impl {
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    use libc::{fdopendir, openat, unlinkat};
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    use libc::{fdopendir, openat64 as openat, unlinkat};
    use super::{Dir, DirEntry, InnerReadDir, ReadDir, lstat};
    use crate::ffi::CStr;
    use crate::io;
    use crate::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd};
    use crate::os::unix::prelude::{OwnedFd, RawFd};
    use crate::path::{Path, PathBuf};
    use crate::sys::common::small_c_string::run_path_with_cstr;
    use crate::sys::{cvt, cvt_r};
    use crate::sys_common::ignore_notfound;
    pub fn openat_nofollow_dironly(parent_fd: Option<RawFd>, p: &CStr) -> io::Result<OwnedFd> {
        let fd = cvt_r(|| unsafe {
            openat(
                parent_fd.unwrap_or(libc::AT_FDCWD),
                p.as_ptr(),
                libc::O_CLOEXEC | libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
            )
        })?;
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
    fn fdreaddir(dir_fd: OwnedFd) -> io::Result<(ReadDir, RawFd)> {
        let ptr = unsafe { fdopendir(dir_fd.as_raw_fd()) };
        if ptr.is_null() {
            return Err(io::Error::last_os_error());
        }
        let dirp = Dir(ptr);
        let new_parent_fd = dir_fd.into_raw_fd();
        let dummy_root = PathBuf::new();
        let inner = InnerReadDir { dirp, root: dummy_root };
        Ok((ReadDir::new(inner), new_parent_fd))
    }
    #[cfg(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks",
        target_os = "aix",
    ))]
    fn is_dir(_ent: &DirEntry) -> Option<bool> {
        None
    }
    #[cfg(not(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks",
        target_os = "aix",
    )))]
    fn is_dir(ent: &DirEntry) -> Option<bool> {
        match ent.entry.d_type {
            libc::DT_UNKNOWN => None,
            libc::DT_DIR => Some(true),
            _ => Some(false),
        }
    }
    fn is_enoent(result: &io::Result<()>) -> bool {
        if let Err(err) = result
            && matches!(err.raw_os_error(), Some(libc::ENOENT))
        {
            true
        } else {
            false
        }
    }
    fn remove_dir_all_recursive(parent_fd: Option<RawFd>, path: &CStr) -> io::Result<()> {
        let fd = match openat_nofollow_dironly(parent_fd, &path) {
            Err(err) if matches!(err.raw_os_error(), Some(libc::ENOTDIR | libc::ELOOP)) => {
                return match parent_fd {
                    Some(parent_fd) => {
                        cvt(unsafe { unlinkat(parent_fd, path.as_ptr(), 0) }).map(drop)
                    }
                    None => Err(err),
                };
            }
            result => result?,
        };
        let (dir, fd) = fdreaddir(fd)?;
        for child in dir {
            let child = child?;
            let child_name = child.name_cstr();
            let result: io::Result<()> = try {
                match is_dir(&child) {
                    Some(true) => {
                        remove_dir_all_recursive(Some(fd), child_name)?;
                    }
                    Some(false) => {
                        cvt(unsafe { unlinkat(fd, child_name.as_ptr(), 0) })?;
                    }
                    None => {
                        remove_dir_all_recursive(Some(fd), child_name)?;
                    }
                }
            };
            if result.is_err() && !is_enoent(&result) {
                return result;
            }
        }
        ignore_notfound(cvt(unsafe {
            unlinkat(parent_fd.unwrap_or(libc::AT_FDCWD), path.as_ptr(), libc::AT_REMOVEDIR)
        }))?;
        Ok(())
    }
    fn remove_dir_all_modern(p: &CStr) -> io::Result<()> {
        let attr = lstat(p)?;
        if attr.file_type().is_symlink() {
            super::unlink(p)
        } else {
            remove_dir_all_recursive(None, &p)
        }
    }
    pub fn remove_dir_all(p: &Path) -> io::Result<()> {
        run_path_with_cstr(p, &remove_dir_all_modern)
    }
}