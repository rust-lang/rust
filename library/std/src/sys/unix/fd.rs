#![unstable(reason = "not public", issue = "none", feature = "fd")]

#[cfg(test)]
mod tests;

use crate::cmp;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, Read};
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::cvt;
use crate::sys_common::{AsInner, FromInner, IntoInner};

#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "emscripten",
    target_os = "l4re"
))]
use libc::off64_t;
#[cfg(not(any(
    target_os = "linux",
    target_os = "emscripten",
    target_os = "l4re",
    target_os = "android"
)))]
use libc::off_t as off64_t;

#[derive(Debug)]
pub struct FileDesc(OwnedFd);

// The maximum read limit on most POSIX-like systems is `SSIZE_MAX`,
// with the man page quoting that if the count of bytes to read is
// greater than `SSIZE_MAX` the result is "unspecified".
//
// On macOS, however, apparently the 64-bit libc is either buggy or
// intentionally showing odd behavior by rejecting any read with a size
// larger than or equal to INT_MAX. To handle both of these the read
// size is capped on both platforms.
#[cfg(target_os = "macos")]
const READ_LIMIT: usize = libc::c_int::MAX as usize - 1;
#[cfg(not(target_os = "macos"))]
const READ_LIMIT: usize = libc::ssize_t::MAX as usize;

#[cfg(any(
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "ios",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "watchos",
))]
const fn max_iov() -> usize {
    libc::IOV_MAX as usize
}

#[cfg(any(
    target_os = "android",
    target_os = "emscripten",
    target_os = "linux",
    target_os = "nto",
))]
const fn max_iov() -> usize {
    libc::UIO_MAXIOV as usize
}

#[cfg(not(any(
    target_os = "android",
    target_os = "dragonfly",
    target_os = "emscripten",
    target_os = "freebsd",
    target_os = "ios",
    target_os = "linux",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "nto",
    target_os = "openbsd",
    target_os = "horizon",
    target_os = "vita",
    target_os = "watchos",
)))]
const fn max_iov() -> usize {
    16 // The minimum value required by POSIX.
}

impl FileDesc {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::read(
                self.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                cmp::min(buf.len(), READ_LIMIT),
            )
        })?;
        Ok(ret as usize)
    }

    #[cfg(not(any(target_os = "espidf", target_os = "horizon", target_os = "vita")))]
    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::readv(
                self.as_raw_fd(),
                bufs.as_mut_ptr() as *mut libc::iovec as *const libc::iovec,
                cmp::min(bufs.len(), max_iov()) as libc::c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita"))]
    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        io::default_read_vectored(|b| self.read(b), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        cfg!(not(any(target_os = "espidf", target_os = "horizon", target_os = "vita")))
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        use libc::pread as pread64;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        use libc::pread64;

        unsafe {
            cvt(pread64(
                self.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                cmp::min(buf.len(), READ_LIMIT),
                offset as off64_t,
            ))
            .map(|n| n as usize)
        }
    }

    pub fn read_buf(&self, mut cursor: BorrowedCursor<'_>) -> io::Result<()> {
        let ret = cvt(unsafe {
            libc::read(
                self.as_raw_fd(),
                cursor.as_mut().as_mut_ptr() as *mut libc::c_void,
                cmp::min(cursor.capacity(), READ_LIMIT),
            )
        })?;

        // Safety: `ret` bytes were written to the initialized portion of the buffer
        unsafe {
            cursor.advance(ret as usize);
        }
        Ok(())
    }

    #[cfg(any(
        target_os = "emscripten",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
    ))]
    pub fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::preadv(
                self.as_raw_fd(),
                bufs.as_mut_ptr() as *mut libc::iovec as *const libc::iovec,
                cmp::min(bufs.len(), max_iov()) as libc::c_int,
                offset as _,
            )
        })?;
        Ok(ret as usize)
    }

    #[cfg(not(any(
        target_os = "android",
        target_os = "emscripten",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "ios",
        target_os = "linux",
        target_os = "macos",
        target_os = "netbsd",
    )))]
    pub fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        io::default_read_vectored(|b| self.read_at(b, offset), bufs)
    }

    // We support some old Android versions that do not have `preadv` in libc,
    // so we use weak linkage and fallback to a direct syscall if not available.
    //
    // On 32-bit targets, we don't want to deal with weird ABI issues around
    // passing 64-bits parameters to syscalls, so we fallback to the default
    // implementation if `preadv` is not available.
    #[cfg(all(target_os = "android", target_pointer_width = "64"))]
    pub fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        super::weak::syscall! {
            fn preadv(
                fd: libc::c_int,
                iovec: *const libc::iovec,
                n_iovec: libc::c_int,
                offset: off64_t
            ) -> isize
        }

        let ret = cvt(unsafe {
            preadv(
                self.as_raw_fd(),
                bufs.as_mut_ptr() as *mut libc::iovec as *const libc::iovec,
                cmp::min(bufs.len(), max_iov()) as libc::c_int,
                offset as _,
            )
        })?;
        Ok(ret as usize)
    }

    // We support old MacOS and iOS versions that do not have `preadv`. There is
    // no `syscall` possible in these platform.
    #[cfg(any(
        all(target_os = "android", target_pointer_width = "32"),
        target_os = "ios",
        target_os = "macos",
    ))]
    pub fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        super::weak::weak!(fn preadv64(libc::c_int, *const libc::iovec, libc::c_int, off64_t) -> isize);

        match preadv64.get() {
            Some(preadv) => {
                let ret = cvt(unsafe {
                    preadv(
                        self.as_raw_fd(),
                        bufs.as_mut_ptr() as *mut libc::iovec as *const libc::iovec,
                        cmp::min(bufs.len(), max_iov()) as libc::c_int,
                        offset as _,
                    )
                })?;
                Ok(ret as usize)
            }
            None => io::default_read_vectored(|b| self.read_at(b, offset), bufs),
        }
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::write(
                self.as_raw_fd(),
                buf.as_ptr() as *const libc::c_void,
                cmp::min(buf.len(), READ_LIMIT),
            )
        })?;
        Ok(ret as usize)
    }

    #[cfg(not(any(target_os = "espidf", target_os = "horizon", target_os = "vita")))]
    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::writev(
                self.as_raw_fd(),
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), max_iov()) as libc::c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita"))]
    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        io::default_write_vectored(|b| self.write(b), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        cfg!(not(any(target_os = "espidf", target_os = "horizon", target_os = "vita")))
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        use libc::pwrite as pwrite64;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        use libc::pwrite64;

        unsafe {
            cvt(pwrite64(
                self.as_raw_fd(),
                buf.as_ptr() as *const libc::c_void,
                cmp::min(buf.len(), READ_LIMIT),
                offset as off64_t,
            ))
            .map(|n| n as usize)
        }
    }

    #[cfg(any(
        target_os = "emscripten",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
    ))]
    pub fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::pwritev(
                self.as_raw_fd(),
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), max_iov()) as libc::c_int,
                offset as _,
            )
        })?;
        Ok(ret as usize)
    }

    #[cfg(not(any(
        target_os = "android",
        target_os = "emscripten",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "ios",
        target_os = "linux",
        target_os = "macos",
        target_os = "netbsd",
    )))]
    pub fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        io::default_write_vectored(|b| self.write_at(b, offset), bufs)
    }

    // We support some old Android versions that do not have `pwritev` in libc,
    // so we use weak linkage and fallback to a direct syscall if not available.
    //
    // On 32-bit targets, we don't want to deal with weird ABI issues around
    // passing 64-bits parameters to syscalls, so we fallback to the default
    // implementation if `pwritev` is not available.
    #[cfg(all(target_os = "android", target_pointer_width = "64"))]
    pub fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        super::weak::syscall! {
            fn pwritev(
                fd: libc::c_int,
                iovec: *const libc::iovec,
                n_iovec: libc::c_int,
                offset: off64_t
            ) -> isize
        }

        let ret = cvt(unsafe {
            pwritev(
                self.as_raw_fd(),
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), max_iov()) as libc::c_int,
                offset as _,
            )
        })?;
        Ok(ret as usize)
    }

    // We support old MacOS and iOS versions that do not have `pwritev`. There is
    // no `syscall` possible in these platform.
    #[cfg(any(
        all(target_os = "android", target_pointer_width = "32"),
        target_os = "ios",
        target_os = "macos",
    ))]
    pub fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        super::weak::weak!(fn pwritev64(libc::c_int, *const libc::iovec, libc::c_int, off64_t) -> isize);

        match pwritev64.get() {
            Some(pwritev) => {
                let ret = cvt(unsafe {
                    pwritev(
                        self.as_raw_fd(),
                        bufs.as_ptr() as *const libc::iovec,
                        cmp::min(bufs.len(), max_iov()) as libc::c_int,
                        offset as _,
                    )
                })?;
                Ok(ret as usize)
            }
            None => io::default_write_vectored(|b| self.write_at(b, offset), bufs),
        }
    }

    #[cfg(not(any(
        target_env = "newlib",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "l4re",
        target_os = "linux",
        target_os = "haiku",
        target_os = "redox",
        target_os = "vxworks",
        target_os = "nto",
    )))]
    pub fn set_cloexec(&self) -> io::Result<()> {
        unsafe {
            cvt(libc::ioctl(self.as_raw_fd(), libc::FIOCLEX))?;
            Ok(())
        }
    }
    #[cfg(any(
        all(target_env = "newlib", not(any(target_os = "espidf", target_os = "horizon"))),
        target_os = "solaris",
        target_os = "illumos",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "l4re",
        target_os = "linux",
        target_os = "haiku",
        target_os = "redox",
        target_os = "vxworks",
        target_os = "nto",
    ))]
    pub fn set_cloexec(&self) -> io::Result<()> {
        unsafe {
            let previous = cvt(libc::fcntl(self.as_raw_fd(), libc::F_GETFD))?;
            let new = previous | libc::FD_CLOEXEC;
            if new != previous {
                cvt(libc::fcntl(self.as_raw_fd(), libc::F_SETFD, new))?;
            }
            Ok(())
        }
    }
    #[cfg(any(target_os = "espidf", target_os = "horizon"))]
    pub fn set_cloexec(&self) -> io::Result<()> {
        // FD_CLOEXEC is not supported in ESP-IDF and Horizon OS but there's no need to,
        // because neither supports spawning processes.
        Ok(())
    }

    #[cfg(target_os = "linux")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        unsafe {
            let v = nonblocking as libc::c_int;
            cvt(libc::ioctl(self.as_raw_fd(), libc::FIONBIO, &v))?;
            Ok(())
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        unsafe {
            let previous = cvt(libc::fcntl(self.as_raw_fd(), libc::F_GETFL))?;
            let new = if nonblocking {
                previous | libc::O_NONBLOCK
            } else {
                previous & !libc::O_NONBLOCK
            };
            if new != previous {
                cvt(libc::fcntl(self.as_raw_fd(), libc::F_SETFL, new))?;
            }
            Ok(())
        }
    }

    #[inline]
    pub fn duplicate(&self) -> io::Result<FileDesc> {
        Ok(Self(self.0.try_clone()?))
    }
}

impl<'a> Read for &'a FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }

    fn read_buf(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        (**self).read_buf(cursor)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        (**self).read_vectored(bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        (**self).is_read_vectored()
    }
}

impl AsInner<OwnedFd> for FileDesc {
    #[inline]
    fn as_inner(&self) -> &OwnedFd {
        &self.0
    }
}

impl IntoInner<OwnedFd> for FileDesc {
    fn into_inner(self) -> OwnedFd {
        self.0
    }
}

impl FromInner<OwnedFd> for FileDesc {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self(owned_fd)
    }
}

impl AsFd for FileDesc {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl AsRawFd for FileDesc {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl IntoRawFd for FileDesc {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for FileDesc {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(FromRawFd::from_raw_fd(raw_fd))
    }
}
