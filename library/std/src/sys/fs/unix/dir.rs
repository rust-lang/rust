use libc::c_int;

cfg_select! {
    not(
        any(
            all(target_os = "linux", not(target_env = "musl")),
            target_os = "l4re",
            target_os = "android",
            target_os = "hurd",
        )
    ) => {
        use libc::{open as open64, openat as openat64};
    }
    _ => {
        use libc::{open64, openat64};
    }
}

use crate::ffi::CStr;
use crate::os::fd::{AsFd, BorrowedFd, IntoRawFd, OwnedFd, RawFd};
#[cfg(target_family = "unix")]
use crate::os::unix::io::{AsRawFd, FromRawFd};
#[cfg(target_os = "wasi")]
use crate::os::wasi::io::{AsRawFd, FromRawFd};
use crate::path::Path;
use crate::sys::fd::FileDesc;
use crate::sys::fs::OpenOptions;
use crate::sys::fs::unix::{File, debug_path_fd};
use crate::sys::helpers::run_path_with_cstr;
use crate::sys::{AsInner, FromInner, IntoInner, cvt_r};
use crate::{fmt, fs, io};

pub struct Dir(OwnedFd);

impl Dir {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<Self> {
        run_path_with_cstr(path, &|path| Self::open_with_c(path, opts))
    }

    pub fn open_file(&self, path: &Path, opts: &OpenOptions) -> io::Result<File> {
        run_path_with_cstr(path.as_ref(), &|path| self.open_file_c(path, &opts))
    }

    pub fn open_with_c(path: &CStr, opts: &OpenOptions) -> io::Result<Self> {
        let flags = libc::O_CLOEXEC
            | libc::O_DIRECTORY
            | opts.get_access_mode()?
            | opts.get_creation_mode()?
            | (opts.custom_flags as c_int & !libc::O_ACCMODE);
        let fd = cvt_r(|| unsafe { open64(path.as_ptr(), flags, opts.mode as c_int) })?;
        Ok(Self(unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    fn open_file_c(&self, path: &CStr, opts: &OpenOptions) -> io::Result<File> {
        let flags = libc::O_CLOEXEC
            | opts.get_access_mode()?
            | opts.get_creation_mode()?
            | (opts.custom_flags as c_int & !libc::O_ACCMODE);
        let fd = cvt_r(|| unsafe {
            openat64(self.0.as_raw_fd(), path.as_ptr(), flags, opts.mode as c_int)
        })?;
        Ok(File(unsafe { FileDesc::from_raw_fd(fd) }))
    }
}

impl fmt::Debug for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fd = self.0.as_raw_fd();
        let mut b = debug_path_fd(fd, f, "Dir");
        b.finish()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl AsRawFd for fs::Dir {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().0.as_raw_fd()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl IntoRawFd for fs::Dir {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().0.into_raw_fd()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl FromRawFd for fs::Dir {
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self::from_inner(Dir(unsafe { FromRawFd::from_raw_fd(fd) }))
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl AsFd for fs::Dir {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().0.as_fd()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl From<fs::Dir> for OwnedFd {
    fn from(value: fs::Dir) -> Self {
        value.into_inner().0
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl From<OwnedFd> for fs::Dir {
    fn from(value: OwnedFd) -> Self {
        Self::from_inner(Dir(value))
    }
}
