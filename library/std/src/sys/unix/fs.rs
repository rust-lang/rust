use crate::os::unix::prelude::*;

use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io::{self, Error, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::sync::Arc;
use crate::sys::fd::FileDesc;
use crate::sys::time::SystemTime;
use crate::sys::{cvt, cvt_r};
use crate::sys_common::{AsInner, FromInner};

use libc::{c_int, mode_t};

#[cfg(any(target_os = "linux", target_os = "emscripten", target_os = "android"))]
use libc::dirfd;
#[cfg(any(target_os = "linux", target_os = "emscripten"))]
use libc::fstatat64;
#[cfg(not(any(
    target_os = "linux",
    target_os = "emscripten",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "l4re",
    target_os = "fuchsia",
    target_os = "redox"
)))]
use libc::readdir_r as readdir64_r;
#[cfg(target_os = "android")]
use libc::{
    dirent as dirent64, fstat as fstat64, fstatat as fstatat64, lseek64, lstat as lstat64,
    open as open64, stat as stat64,
};
#[cfg(not(any(
    target_os = "linux",
    target_os = "emscripten",
    target_os = "l4re",
    target_os = "android"
)))]
use libc::{
    dirent as dirent64, fstat as fstat64, ftruncate as ftruncate64, lseek as lseek64,
    lstat as lstat64, off_t as off64_t, open as open64, stat as stat64,
};
#[cfg(any(target_os = "linux", target_os = "emscripten", target_os = "l4re"))]
use libc::{
    dirent64, fstat64, ftruncate64, lseek64, lstat64, off64_t, open64, readdir64_r, stat64,
};

pub use crate::sys_common::fs::{remove_dir_all, try_exists};

pub struct File(FileDesc);

// FIXME: This should be available on Linux with all `target_env`.
// But currently only glibc exposes `statx` fn and structs.
// We don't want to import unverified raw C structs here directly.
// https://github.com/rust-lang/rust/pull/67774
macro_rules! cfg_has_statx {
    ({ $($then_tt:tt)* } else { $($else_tt:tt)* }) => {
        cfg_if::cfg_if! {
            if #[cfg(all(target_os = "linux", target_env = "gnu"))] {
                $($then_tt)*
            } else {
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
        // This is needed to check if btime is supported by the filesystem.
        stx_mask: u32,
        stx_btime: libc::statx_timestamp,
    }

    // We prefer `statx` on Linux if available, which contains file creation time.
    // Default `stat64` contains no creation time.
    unsafe fn try_statx(
        fd: c_int,
        path: *const libc::c_char,
        flags: i32,
        mask: u32,
    ) -> Option<io::Result<FileAttr>> {
        use crate::sync::atomic::{AtomicU8, Ordering};

        // Linux kernel prior to 4.11 or glibc prior to glibc 2.28 don't support `statx`
        // We store the availability in global to avoid unnecessary syscalls.
        // 0: Unknown
        // 1: Not available
        // 2: Available
        static STATX_STATE: AtomicU8 = AtomicU8::new(0);
        syscall! {
            fn statx(
                fd: c_int,
                pathname: *const libc::c_char,
                flags: c_int,
                mask: libc::c_uint,
                statxbuf: *mut libc::statx
            ) -> c_int
        }

        match STATX_STATE.load(Ordering::Relaxed) {
            0 => {
                // It is a trick to call `statx` with null pointers to check if the syscall
                // is available. According to the manual, it is expected to fail with EFAULT.
                // We do this mainly for performance, since it is nearly hundreds times
                // faster than a normal successful call.
                let err = cvt(statx(0, ptr::null(), 0, libc::STATX_ALL, ptr::null_mut()))
                    .err()
                    .and_then(|e| e.raw_os_error());
                // We don't check `err == Some(libc::ENOSYS)` because the syscall may be limited
                // and returns `EPERM`. Listing all possible errors seems not a good idea.
                // See: https://github.com/rust-lang/rust/issues/65662
                if err != Some(libc::EFAULT) {
                    STATX_STATE.store(1, Ordering::Relaxed);
                    return None;
                }
                STATX_STATE.store(2, Ordering::Relaxed);
            }
            1 => return None,
            _ => {}
        }

        let mut buf: libc::statx = mem::zeroed();
        if let Err(err) = cvt(statx(fd, path, flags, mask, &mut buf)) {
            return Some(Err(err));
        }

        // We cannot fill `stat64` exhaustively because of private padding fields.
        let mut stat: stat64 = mem::zeroed();
        // `c_ulong` on gnu-mips, `dev_t` otherwise
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
        // `i64` on gnu-x86_64-x32, `c_ulong` otherwise.
        stat.st_atime_nsec = buf.stx_atime.tv_nsec as _;
        stat.st_mtime = buf.stx_mtime.tv_sec as libc::time_t;
        stat.st_mtime_nsec = buf.stx_mtime.tv_nsec as _;
        stat.st_ctime = buf.stx_ctime.tv_sec as libc::time_t;
        stat.st_ctime_nsec = buf.stx_ctime.tv_nsec as _;

        let extra = StatxExtraFields {
            stx_mask: buf.stx_mask,
            stx_btime: buf.stx_btime,
        };

        Some(Ok(FileAttr { stat, statx_extra_fields: Some(extra) }))
    }

} else {
    #[derive(Clone)]
    pub struct FileAttr {
        stat: stat64,
    }
}}

// all DirEntry's will have a reference to this struct
struct InnerReadDir {
    dirp: Dir,
    root: PathBuf,
}

pub struct ReadDir {
    inner: Arc<InnerReadDir>,
    #[cfg(not(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "fuchsia",
        target_os = "redox",
    )))]
    end_of_stream: bool,
}

struct Dir(*mut libc::DIR);

unsafe impl Send for Dir {}
unsafe impl Sync for Dir {}

pub struct DirEntry {
    entry: dirent64,
    dir: Arc<InnerReadDir>,
    // We need to store an owned copy of the entry name
    // on Solaris and Fuchsia because a) it uses a zero-length
    // array to store the name, b) its lifetime between readdir
    // calls is not guaranteed.
    #[cfg(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "fuchsia",
        target_os = "redox"
    ))]
    name: Box<[u8]>,
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
    custom_flags: i32,
    mode: mode_t,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    mode: mode_t,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType {
    mode: mode_t,
}

#[derive(Debug)]
pub struct DirBuilder {
    mode: mode_t,
}

cfg_has_statx! {{
    impl FileAttr {
        fn from_stat64(stat: stat64) -> Self {
            Self { stat, statx_extra_fields: None }
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
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_mtime as libc::time_t,
            tv_nsec: self.stat.st_mtimensec as libc::c_long,
        }))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_atime as libc::time_t,
            tv_nsec: self.stat.st_atimensec as libc::c_long,
        }))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_birthtime as libc::time_t,
            tv_nsec: self.stat.st_birthtimensec as libc::c_long,
        }))
    }
}

#[cfg(not(target_os = "netbsd"))]
impl FileAttr {
    #[cfg(not(target_os = "vxworks"))]
    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_mtime as libc::time_t,
            tv_nsec: self.stat.st_mtime_nsec as _,
        }))
    }

    #[cfg(target_os = "vxworks")]
    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_mtime as libc::time_t,
            tv_nsec: 0,
        }))
    }

    #[cfg(not(target_os = "vxworks"))]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_atime as libc::time_t,
            tv_nsec: self.stat.st_atime_nsec as _,
        }))
    }

    #[cfg(target_os = "vxworks")]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_atime as libc::time_t,
            tv_nsec: 0,
        }))
    }

    #[cfg(any(
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "macos",
        target_os = "ios"
    ))]
    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_birthtime as libc::time_t,
            tv_nsec: self.stat.st_birthtime_nsec as libc::c_long,
        }))
    }

    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "macos",
        target_os = "ios"
    )))]
    pub fn created(&self) -> io::Result<SystemTime> {
        cfg_has_statx! {
            if let Some(ext) = &self.statx_extra_fields {
                return if (ext.stx_mask & libc::STATX_BTIME) != 0 {
                    Ok(SystemTime::from(libc::timespec {
                        tv_sec: ext.stx_btime.tv_sec as libc::time_t,
                        tv_nsec: ext.stx_btime.tv_nsec as _,
                    }))
                } else {
                    Err(io::Error::new_const(
                        io::ErrorKind::Uncategorized,
                        &"creation time is not available for the filesystem",
                    ))
                };
            }
        }

        Err(io::Error::new_const(
            io::ErrorKind::Unsupported,
            &"creation time is not available on this platform \
                            currently",
        ))
    }
}

impl AsInner<stat64> for FileAttr {
    fn as_inner(&self) -> &stat64 {
        &self.stat
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        // check if any class (owner, group, others) has write permission
        self.mode & 0o222 == 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            // remove write permission for all classes; equivalent to `chmod a-w <file>`
            self.mode &= !0o222;
        } else {
            // add write permission for all classes; equivalent to `chmod a+w <file>`
            self.mode |= 0o222;
        }
    }
    pub fn mode(&self) -> u32 {
        self.mode as u32
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
        self.mode & libc::S_IFMT == mode
    }
}

impl FromInner<u32> for FilePermissions {
    fn from_inner(mode: u32) -> FilePermissions {
        FilePermissions { mode: mode as mode_t }
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

    #[cfg(any(
        target_os = "solaris",
        target_os = "fuchsia",
        target_os = "redox",
        target_os = "illumos"
    ))]
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        use crate::slice;

        unsafe {
            loop {
                // Although readdir_r(3) would be a correct function to use here because
                // of the thread safety, on Illumos and Fuchsia the readdir(3C) function
                // is safe to use in threaded applications and it is generally preferred
                // over the readdir_r(3C) function.
                super::os::set_errno(0);
                let entry_ptr = libc::readdir(self.inner.dirp.0);
                if entry_ptr.is_null() {
                    // null can mean either the end is reached or an error occurred.
                    // So we had to clear errno beforehand to check for an error now.
                    return match super::os::errno() {
                        0 => None,
                        e => Some(Err(Error::from_raw_os_error(e))),
                    };
                }

                let name = (*entry_ptr).d_name.as_ptr();
                let namelen = libc::strlen(name) as usize;

                let ret = DirEntry {
                    entry: *entry_ptr,
                    name: slice::from_raw_parts(name as *const u8, namelen as usize)
                        .to_owned()
                        .into_boxed_slice(),
                    dir: Arc::clone(&self.inner),
                };
                if ret.name_bytes() != b"." && ret.name_bytes() != b".." {
                    return Some(Ok(ret));
                }
            }
        }
    }

    #[cfg(not(any(
        target_os = "solaris",
        target_os = "fuchsia",
        target_os = "redox",
        target_os = "illumos"
    )))]
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        if self.end_of_stream {
            return None;
        }

        unsafe {
            let mut ret = DirEntry { entry: mem::zeroed(), dir: Arc::clone(&self.inner) };
            let mut entry_ptr = ptr::null_mut();
            loop {
                if readdir64_r(self.inner.dirp.0, &mut ret.entry, &mut entry_ptr) != 0 {
                    if entry_ptr.is_null() {
                        // We encountered an error (which will be returned in this iteration), but
                        // we also reached the end of the directory stream. The `end_of_stream`
                        // flag is enabled to make sure that we return `None` in the next iteration
                        // (instead of looping forever)
                        self.end_of_stream = true;
                    }
                    return Some(Err(Error::last_os_error()));
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

impl Drop for Dir {
    fn drop(&mut self) {
        let r = unsafe { libc::closedir(self.0) };
        debug_assert_eq!(r, 0);
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.dir.root.join(OsStr::from_bytes(self.name_bytes()))
    }

    pub fn file_name(&self) -> OsString {
        OsStr::from_bytes(self.name_bytes()).to_os_string()
    }

    #[cfg(any(target_os = "linux", target_os = "emscripten", target_os = "android"))]
    pub fn metadata(&self) -> io::Result<FileAttr> {
        let fd = cvt(unsafe { dirfd(self.dir.dirp.0) })?;
        let name = self.entry.d_name.as_ptr();

        cfg_has_statx! {
            if let Some(ret) = unsafe { try_statx(
                fd,
                name,
                libc::AT_SYMLINK_NOFOLLOW | libc::AT_STATX_SYNC_AS_STAT,
                libc::STATX_ALL,
            ) } {
                return ret;
            }
        }

        let mut stat: stat64 = unsafe { mem::zeroed() };
        cvt(unsafe { fstatat64(fd, name, &mut stat, libc::AT_SYMLINK_NOFOLLOW) })?;
        Ok(FileAttr::from_stat64(stat))
    }

    #[cfg(not(any(target_os = "linux", target_os = "emscripten", target_os = "android")))]
    pub fn metadata(&self) -> io::Result<FileAttr> {
        lstat(&self.path())
    }

    #[cfg(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks"
    ))]
    pub fn file_type(&self) -> io::Result<FileType> {
        lstat(&self.path()).map(|m| m.file_type())
    }

    #[cfg(not(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks"
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
            _ => lstat(&self.path()).map(|m| m.file_type()),
        }
    }

    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "emscripten",
        target_os = "android",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "l4re",
        target_os = "fuchsia",
        target_os = "redox",
        target_os = "vxworks"
    ))]
    pub fn ino(&self) -> u64 {
        self.entry.d_ino as u64
    }

    #[cfg(any(
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))]
    pub fn ino(&self) -> u64 {
        self.entry.d_fileno as u64
    }

    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "freebsd",
        target_os = "dragonfly"
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
    #[cfg(any(
        target_os = "android",
        target_os = "linux",
        target_os = "emscripten",
        target_os = "l4re",
        target_os = "haiku",
        target_os = "vxworks"
    ))]
    fn name_bytes(&self) -> &[u8] {
        unsafe { CStr::from_ptr(self.entry.d_name.as_ptr()).to_bytes() }
    }
    #[cfg(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "fuchsia",
        target_os = "redox"
    ))]
    fn name_bytes(&self) -> &[u8] {
        &*self.name
    }

    pub fn file_name_os_str(&self) -> &OsStr {
        OsStr::from_bytes(self.name_bytes())
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
            (false, false, false) => Err(Error::from_raw_os_error(libc::EINVAL)),
        }
    }

    fn get_creation_mode(&self) -> io::Result<c_int> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(Error::from_raw_os_error(libc::EINVAL));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(Error::from_raw_os_error(libc::EINVAL));
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

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = cstr(path)?;
        File::open_c(&path, opts)
    }

    pub fn open_c(path: &CStr, opts: &OpenOptions) -> io::Result<File> {
        let flags = libc::O_CLOEXEC
            | opts.get_access_mode()?
            | opts.get_creation_mode()?
            | (opts.custom_flags as c_int & !libc::O_ACCMODE);
        // The third argument of `open64` is documented to have type `mode_t`. On
        // some platforms (like macOS, where `open64` is actually `open`), `mode_t` is `u16`.
        // However, since this is a variadic function, C integer promotion rules mean that on
        // the ABI level, this still gets passed as `c_int` (aka `u32` on Unix platforms).
        let fd = cvt_r(|| unsafe { open64(path.as_ptr(), flags, opts.mode as c_int) })?;
        Ok(File(FileDesc::new(fd)))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let fd = self.0.raw();

        cfg_has_statx! {
            if let Some(ret) = unsafe { try_statx(
                fd,
                b"\0" as *const _ as *const libc::c_char,
                libc::AT_EMPTY_PATH | libc::AT_STATX_SYNC_AS_STAT,
                libc::STATX_ALL,
            ) } {
                return ret;
            }
        }

        let mut stat: stat64 = unsafe { mem::zeroed() };
        cvt(unsafe { fstat64(fd, &mut stat) })?;
        Ok(FileAttr::from_stat64(stat))
    }

    pub fn fsync(&self) -> io::Result<()> {
        cvt_r(|| unsafe { os_fsync(self.0.raw()) })?;
        return Ok(());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe fn os_fsync(fd: c_int) -> c_int {
            libc::fcntl(fd, libc::F_FULLFSYNC)
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        unsafe fn os_fsync(fd: c_int) -> c_int {
            libc::fsync(fd)
        }
    }

    pub fn datasync(&self) -> io::Result<()> {
        cvt_r(|| unsafe { os_datasync(self.0.raw()) })?;
        return Ok(());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fcntl(fd, libc::F_FULLFSYNC)
        }
        #[cfg(any(
            target_os = "freebsd",
            target_os = "linux",
            target_os = "android",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fdatasync(fd)
        }
        #[cfg(not(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "netbsd",
            target_os = "openbsd"
        )))]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fsync(fd)
        }
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        #[cfg(target_os = "android")]
        return crate::sys::android::ftruncate64(self.0.raw(), size);

        #[cfg(not(target_os = "android"))]
        {
            use crate::convert::TryInto;
            let size: off64_t =
                size.try_into().map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            cvt_r(|| unsafe { ftruncate64(self.0.raw(), size) }).map(drop)
        }
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

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, too large values will end up as
            // negative which will cause an error in `lseek64`.
            SeekFrom::Start(off) => (libc::SEEK_SET, off as i64),
            SeekFrom::End(off) => (libc::SEEK_END, off),
            SeekFrom::Current(off) => (libc::SEEK_CUR, off),
        };
        let n = cvt(unsafe { lseek64(self.0.raw(), pos, whence) })?;
        Ok(n as u64)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        self.0.duplicate().map(File)
    }

    pub fn fd(&self) -> &FileDesc {
        &self.0
    }

    pub fn into_fd(self) -> FileDesc {
        self.0
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        cvt_r(|| unsafe { libc::fchmod(self.0.raw(), perm.mode) })?;
        Ok(())
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder { mode: 0o777 }
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let p = cstr(p)?;
        cvt(unsafe { libc::mkdir(p.as_ptr(), self.mode) })?;
        Ok(())
    }

    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode as mode_t;
    }
}

fn cstr(path: &Path) -> io::Result<CString> {
    Ok(CString::new(path.as_os_str().as_bytes())?)
}

impl FromInner<c_int> for File {
    fn from_inner(fd: c_int) -> File {
        File(FileDesc::new(fd))
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(target_os = "linux")]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            let mut p = PathBuf::from("/proc/self/fd");
            p.push(&fd.to_string());
            readlink(&p).ok()
        }

        #[cfg(target_os = "macos")]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            // FIXME: The use of PATH_MAX is generally not encouraged, but it
            // is inevitable in this case because macOS defines `fcntl` with
            // `F_GETPATH` in terms of `MAXPATHLEN`, and there are no
            // alternatives. If a better method is invented, it should be used
            // instead.
            let mut buf = vec![0; libc::PATH_MAX as usize];
            let n = unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_ptr()) };
            if n == -1 {
                return None;
            }
            let l = buf.iter().position(|&c| c == 0).unwrap();
            buf.truncate(l as usize);
            buf.shrink_to_fit();
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

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "vxworks")))]
        fn get_path(_fd: c_int) -> Option<PathBuf> {
            // FIXME(#24570): implement this for other Unix platforms
            None
        }

        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "vxworks"))]
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

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "vxworks")))]
        fn get_mode(_fd: c_int) -> Option<(bool, bool)> {
            // FIXME(#24570): implement this for other Unix platforms
            None
        }

        let fd = self.0.raw();
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

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = p.to_path_buf();
    let p = cstr(p)?;
    unsafe {
        let ptr = libc::opendir(p.as_ptr());
        if ptr.is_null() {
            Err(Error::last_os_error())
        } else {
            let inner = InnerReadDir { dirp: Dir(ptr), root };
            Ok(ReadDir {
                inner: Arc::new(inner),
                #[cfg(not(any(
                    target_os = "solaris",
                    target_os = "illumos",
                    target_os = "fuchsia",
                    target_os = "redox",
                )))]
                end_of_stream: false,
            })
        }
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let p = cstr(p)?;
    cvt(unsafe { libc::unlink(p.as_ptr()) })?;
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let old = cstr(old)?;
    let new = cstr(new)?;
    cvt(unsafe { libc::rename(old.as_ptr(), new.as_ptr()) })?;
    Ok(())
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let p = cstr(p)?;
    cvt_r(|| unsafe { libc::chmod(p.as_ptr(), perm.mode) })?;
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = cstr(p)?;
    cvt(unsafe { libc::rmdir(p.as_ptr()) })?;
    Ok(())
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let c_path = cstr(p)?;
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

        // Trigger the internal buffer resizing logic of `Vec` by requiring
        // more space than the current capacity. The length is guaranteed to be
        // the same as the capacity due to the if statement above.
        buf.reserve(1);
    }
}

pub fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    let original = cstr(original)?;
    let link = cstr(link)?;
    cvt(unsafe { libc::symlink(original.as_ptr(), link.as_ptr()) })?;
    Ok(())
}

pub fn link(original: &Path, link: &Path) -> io::Result<()> {
    let original = cstr(original)?;
    let link = cstr(link)?;
    cfg_if::cfg_if! {
        if #[cfg(any(target_os = "vxworks", target_os = "redox", target_os = "android"))] {
            // VxWorks, Redox, and old versions of Android lack `linkat`, so use
            // `link` instead. POSIX leaves it implementation-defined whether
            // `link` follows symlinks, so rely on the `symlink_hard_link` test
            // in library/std/src/fs/tests.rs to check the behavior.
            cvt(unsafe { libc::link(original.as_ptr(), link.as_ptr()) })?;
        } else {
            // Use `linkat` with `AT_FDCWD` instead of `link` as `linkat` gives
            // us a flag to specify how symlinks should be handled. Pass 0 as
            // the flags argument, meaning don't follow symlinks.
            cvt(unsafe { libc::linkat(libc::AT_FDCWD, original.as_ptr(), libc::AT_FDCWD, link.as_ptr(), 0) })?;
        }
    }
    Ok(())
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let p = cstr(p)?;

    cfg_has_statx! {
        if let Some(ret) = unsafe { try_statx(
            libc::AT_FDCWD,
            p.as_ptr(),
            libc::AT_STATX_SYNC_AS_STAT,
            libc::STATX_ALL,
        ) } {
            return ret;
        }
    }

    let mut stat: stat64 = unsafe { mem::zeroed() };
    cvt(unsafe { stat64(p.as_ptr(), &mut stat) })?;
    Ok(FileAttr::from_stat64(stat))
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let p = cstr(p)?;

    cfg_has_statx! {
        if let Some(ret) = unsafe { try_statx(
            libc::AT_FDCWD,
            p.as_ptr(),
            libc::AT_SYMLINK_NOFOLLOW | libc::AT_STATX_SYNC_AS_STAT,
            libc::STATX_ALL,
        ) } {
            return ret;
        }
    }

    let mut stat: stat64 = unsafe { mem::zeroed() };
    cvt(unsafe { lstat64(p.as_ptr(), &mut stat) })?;
    Ok(FileAttr::from_stat64(stat))
}

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    let path = CString::new(p.as_os_str().as_bytes())?;
    let buf;
    unsafe {
        let r = libc::realpath(path.as_ptr(), ptr::null_mut());
        if r.is_null() {
            return Err(io::Error::last_os_error());
        }
        buf = CStr::from_ptr(r).to_bytes().to_vec();
        libc::free(r as *mut _);
    }
    Ok(PathBuf::from(OsString::from_vec(buf)))
}

fn open_from(from: &Path) -> io::Result<(crate::fs::File, crate::fs::Metadata)> {
    use crate::fs::File;
    use crate::sys_common::fs::NOT_FILE_ERROR;

    let reader = File::open(from)?;
    let metadata = reader.metadata()?;
    if !metadata.is_file() {
        return Err(NOT_FILE_ERROR);
    }
    Ok((reader, metadata))
}

fn open_to_and_set_permissions(
    to: &Path,
    reader_metadata: crate::fs::Metadata,
) -> io::Result<(crate::fs::File, crate::fs::Metadata)> {
    use crate::fs::OpenOptions;
    use crate::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let perm = reader_metadata.permissions();
    let writer = OpenOptions::new()
        // create the file with the correct mode right away
        .mode(perm.mode())
        .write(true)
        .create(true)
        .truncate(true)
        .open(to)?;
    let writer_metadata = writer.metadata()?;
    if writer_metadata.is_file() {
        // Set the correct file permissions, in case the file already existed.
        // Don't set the permissions on already existing non-files like
        // pipes/FIFOs or device nodes.
        writer.set_permissions(perm)?;
    }
    Ok((writer, writer_metadata))
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    let (mut reader, reader_metadata) = open_from(from)?;
    let (mut writer, _) = open_to_and_set_permissions(to, reader_metadata)?;

    io::copy(&mut reader, &mut writer)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    let (mut reader, reader_metadata) = open_from(from)?;
    let max_len = u64::MAX;
    let (mut writer, _) = open_to_and_set_permissions(to, reader_metadata)?;

    use super::kernel_copy::{copy_regular_files, CopyResult};

    match copy_regular_files(reader.as_raw_fd(), writer.as_raw_fd(), max_len) {
        CopyResult::Ended(bytes) => Ok(bytes),
        CopyResult::Error(e, _) => Err(e),
        CopyResult::Fallback(written) => match io::copy::generic_copy(&mut reader, &mut writer) {
            Ok(bytes) => Ok(bytes + written),
            Err(e) => Err(e),
        },
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use crate::sync::atomic::{AtomicBool, Ordering};

    const COPYFILE_ACL: u32 = 1 << 0;
    const COPYFILE_STAT: u32 = 1 << 1;
    const COPYFILE_XATTR: u32 = 1 << 2;
    const COPYFILE_DATA: u32 = 1 << 3;

    const COPYFILE_SECURITY: u32 = COPYFILE_STAT | COPYFILE_ACL;
    const COPYFILE_METADATA: u32 = COPYFILE_SECURITY | COPYFILE_XATTR;
    const COPYFILE_ALL: u32 = COPYFILE_METADATA | COPYFILE_DATA;

    const COPYFILE_STATE_COPIED: u32 = 8;

    #[allow(non_camel_case_types)]
    type copyfile_state_t = *mut libc::c_void;
    #[allow(non_camel_case_types)]
    type copyfile_flags_t = u32;

    extern "C" {
        fn fcopyfile(
            from: libc::c_int,
            to: libc::c_int,
            state: copyfile_state_t,
            flags: copyfile_flags_t,
        ) -> libc::c_int;
        fn copyfile_state_alloc() -> copyfile_state_t;
        fn copyfile_state_free(state: copyfile_state_t) -> libc::c_int;
        fn copyfile_state_get(
            state: copyfile_state_t,
            flag: u32,
            dst: *mut libc::c_void,
        ) -> libc::c_int;
    }

    struct FreeOnDrop(copyfile_state_t);
    impl Drop for FreeOnDrop {
        fn drop(&mut self) {
            // The code below ensures that `FreeOnDrop` is never a null pointer
            unsafe {
                // `copyfile_state_free` returns -1 if the `to` or `from` files
                // cannot be closed. However, this is not considered this an
                // error.
                copyfile_state_free(self.0);
            }
        }
    }

    // MacOS prior to 10.12 don't support `fclonefileat`
    // We store the availability in a global to avoid unnecessary syscalls
    static HAS_FCLONEFILEAT: AtomicBool = AtomicBool::new(true);
    syscall! {
        fn fclonefileat(
            srcfd: libc::c_int,
            dst_dirfd: libc::c_int,
            dst: *const libc::c_char,
            flags: libc::c_int
        ) -> libc::c_int
    }

    let (reader, reader_metadata) = open_from(from)?;

    // Opportunistically attempt to create a copy-on-write clone of `from`
    // using `fclonefileat`.
    if HAS_FCLONEFILEAT.load(Ordering::Relaxed) {
        let to = cstr(to)?;
        let clonefile_result =
            cvt(unsafe { fclonefileat(reader.as_raw_fd(), libc::AT_FDCWD, to.as_ptr(), 0) });
        match clonefile_result {
            Ok(_) => return Ok(reader_metadata.len()),
            Err(err) => match err.raw_os_error() {
                // `fclonefileat` will fail on non-APFS volumes, if the
                // destination already exists, or if the source and destination
                // are on different devices. In all these cases `fcopyfile`
                // should succeed.
                Some(libc::ENOTSUP) | Some(libc::EEXIST) | Some(libc::EXDEV) => (),
                Some(libc::ENOSYS) => HAS_FCLONEFILEAT.store(false, Ordering::Relaxed),
                _ => return Err(err),
            },
        }
    }

    // Fall back to using `fcopyfile` if `fclonefileat` does not succeed.
    let (writer, writer_metadata) = open_to_and_set_permissions(to, reader_metadata)?;

    // We ensure that `FreeOnDrop` never contains a null pointer so it is
    // always safe to call `copyfile_state_free`
    let state = unsafe {
        let state = copyfile_state_alloc();
        if state.is_null() {
            return Err(crate::io::Error::last_os_error());
        }
        FreeOnDrop(state)
    };

    let flags = if writer_metadata.is_file() { COPYFILE_ALL } else { COPYFILE_DATA };

    cvt(unsafe { fcopyfile(reader.as_raw_fd(), writer.as_raw_fd(), state.0, flags) })?;

    let mut bytes_copied: libc::off_t = 0;
    cvt(unsafe {
        copyfile_state_get(
            state.0,
            COPYFILE_STATE_COPIED,
            &mut bytes_copied as *mut libc::off_t as *mut libc::c_void,
        )
    })?;
    Ok(bytes_copied as u64)
}

#[cfg(not(any(target_os = "fuchsia", target_os = "vxworks")))]
pub fn chroot(dir: &Path) -> io::Result<()> {
    let dir = cstr(dir)?;
    cvt(unsafe { libc::chroot(dir.as_ptr()) })?;
    Ok(())
}
