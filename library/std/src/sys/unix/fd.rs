#![unstable(reason = "not public", issue = "none", feature = "fd")]

use crate::cmp;
use crate::io::{self, Initializer, IoSlice, IoSliceMut, Read};
use crate::mem;
use crate::sys::cvt;
use crate::sys_common::AsInner;

use libc::{c_int, c_void};

// Android's libc allows for file descriptors to be marked with a tag that marks ownership.
// If a tagged file descriptor is closed with the wrong tag, either a backtrace is printed to logs,
// or the process aborts, depending on process configuration.
#[cfg(target_os = "android")]
mod android {
    use crate::sync::atomic::{AtomicU64, Ordering};
    use libc::c_int;

    #[derive(Debug)]
    pub struct CloseTag(u64);

    static FD_TAG: AtomicU64 = AtomicU64::new(0);

    fn next_tag() -> u64 {
        // The most significant 8 bits of the tag are used to identify the type of the owner, as
        // a debugging aid. The value 13 has been reserved to identify Rust-owned fds.
        let tag = FD_TAG.fetch_add(1, Ordering::Relaxed);
        tag | 13 << 56
    }

    impl CloseTag {
        pub fn tag(fd: c_int) -> CloseTag {
            weak!(fn android_fdsan_exchange_owner_tag(c_int, u64, u64) -> u64);
            match android_fdsan_exchange_owner_tag.get() {
                Some(f) => {
                    let tag = next_tag();
                    let prev = unsafe { f(fd, 0, tag) };
                    if prev != 0 {
                        panic!("attempted to take ownership of an already-owned file descriptor");
                    }
                    CloseTag(tag)
                }

                None => CloseTag(0),
            }
        }

        pub fn close(&mut self, fd: c_int) {
            weak!(fn android_fdsan_close_with_tag(c_int, u64) -> c_int);
            match android_fdsan_close_with_tag.get() {
                Some(f) => unsafe {
                    f(fd, self.0);
                },

                None => {
                    assert_eq!(self.0, 0);
                    unsafe {
                        libc::close(fd);
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct FileDesc {
    fd: c_int,
    #[cfg(target_os = "android")]
    tag: android::CloseTag,
}

// The maximum read limit on most POSIX-like systems is `SSIZE_MAX`,
// with the man page quoting that if the count of bytes to read is
// greater than `SSIZE_MAX` the result is "unspecified".
//
// On macOS, however, apparently the 64-bit libc is either buggy or
// intentionally showing odd behavior by rejecting any read with a size
// larger than or equal to INT_MAX. To handle both of these the read
// size is capped on both platforms.
#[cfg(target_os = "macos")]
const READ_LIMIT: usize = c_int::MAX as usize - 1;
#[cfg(not(target_os = "macos"))]
const READ_LIMIT: usize = libc::ssize_t::MAX as usize;

impl FileDesc {
    #[cfg(target_os = "android")]
    pub fn new(fd: c_int) -> FileDesc {
        FileDesc { fd, tag: android::CloseTag::tag(fd) }
    }

    #[cfg(not(target_os = "android"))]
    pub fn new(fd: c_int) -> FileDesc {
        FileDesc { fd }
    }

    pub fn raw(&self) -> c_int {
        self.fd
    }

    /// Extracts the actual file descriptor without closing it.
    pub fn into_raw(self) -> c_int {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::read(self.fd, buf.as_mut_ptr() as *mut c_void, cmp::min(buf.len(), READ_LIMIT))
        })?;
        Ok(ret as usize)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::readv(
                self.fd,
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), c_int::MAX as usize) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        #[cfg(target_os = "android")]
        use super::android::cvt_pread64;

        #[cfg(not(target_os = "android"))]
        unsafe fn cvt_pread64(
            fd: c_int,
            buf: *mut c_void,
            count: usize,
            offset: i64,
        ) -> io::Result<isize> {
            #[cfg(not(target_os = "linux"))]
            use libc::pread as pread64;
            #[cfg(target_os = "linux")]
            use libc::pread64;
            cvt(pread64(fd, buf, count, offset))
        }

        unsafe {
            cvt_pread64(
                self.fd,
                buf.as_mut_ptr() as *mut c_void,
                cmp::min(buf.len(), READ_LIMIT),
                offset as i64,
            )
            .map(|n| n as usize)
        }
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::write(self.fd, buf.as_ptr() as *const c_void, cmp::min(buf.len(), READ_LIMIT))
        })?;
        Ok(ret as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::writev(
                self.fd,
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), c_int::MAX as usize) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        #[cfg(target_os = "android")]
        use super::android::cvt_pwrite64;

        #[cfg(not(target_os = "android"))]
        unsafe fn cvt_pwrite64(
            fd: c_int,
            buf: *const c_void,
            count: usize,
            offset: i64,
        ) -> io::Result<isize> {
            #[cfg(not(target_os = "linux"))]
            use libc::pwrite as pwrite64;
            #[cfg(target_os = "linux")]
            use libc::pwrite64;
            cvt(pwrite64(fd, buf, count, offset))
        }

        unsafe {
            cvt_pwrite64(
                self.fd,
                buf.as_ptr() as *const c_void,
                cmp::min(buf.len(), READ_LIMIT),
                offset as i64,
            )
            .map(|n| n as usize)
        }
    }

    #[cfg(target_os = "linux")]
    pub fn get_cloexec(&self) -> io::Result<bool> {
        unsafe { Ok((cvt(libc::fcntl(self.fd, libc::F_GETFD))? & libc::FD_CLOEXEC) != 0) }
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
        target_os = "redox"
    )))]
    pub fn set_cloexec(&self) -> io::Result<()> {
        unsafe {
            cvt(libc::ioctl(self.fd, libc::FIOCLEX))?;
            Ok(())
        }
    }
    #[cfg(any(
        target_env = "newlib",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "l4re",
        target_os = "linux",
        target_os = "haiku",
        target_os = "redox"
    ))]
    pub fn set_cloexec(&self) -> io::Result<()> {
        unsafe {
            let previous = cvt(libc::fcntl(self.fd, libc::F_GETFD))?;
            let new = previous | libc::FD_CLOEXEC;
            if new != previous {
                cvt(libc::fcntl(self.fd, libc::F_SETFD, new))?;
            }
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        unsafe {
            let v = nonblocking as c_int;
            cvt(libc::ioctl(self.fd, libc::FIONBIO, &v))?;
            Ok(())
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        unsafe {
            let previous = cvt(libc::fcntl(self.fd, libc::F_GETFL))?;
            let new = if nonblocking {
                previous | libc::O_NONBLOCK
            } else {
                previous & !libc::O_NONBLOCK
            };
            if new != previous {
                cvt(libc::fcntl(self.fd, libc::F_SETFL, new))?;
            }
            Ok(())
        }
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        // We want to atomically duplicate this file descriptor and set the
        // CLOEXEC flag, and currently that's done via F_DUPFD_CLOEXEC. This
        // is a POSIX flag that was added to Linux in 2.6.24.
        let fd = cvt(unsafe { libc::fcntl(self.raw(), libc::F_DUPFD_CLOEXEC, 0) })?;
        Ok(FileDesc::new(fd))
    }
}

impl<'a> Read for &'a FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

impl AsInner<c_int> for FileDesc {
    fn as_inner(&self) -> &c_int {
        &self.fd
    }
}

impl Drop for FileDesc {
    #[cfg(target_os = "android")]
    fn drop(&mut self) {
        self.tag.close(self.fd);
    }

    #[cfg(not(target_os = "android"))]
    fn drop(&mut self) {
        // Note that errors are ignored when closing a file descriptor. The
        // reason for this is that if an error occurs we don't actually know if
        // the file descriptor was closed or not, and if we retried (for
        // something like EINTR), we might close another valid file descriptor
        // opened after we closed ours.
        let _ = unsafe { libc::close(self.fd) };
    }
}
