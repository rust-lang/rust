#![unstable(reason = "not public", issue = "none", feature = "fd")]

use crate::cmp;
use crate::io::{self, Initializer, IoSlice, IoSliceMut, Read};
use crate::mem;
use crate::sys::cvt;
use crate::sys_common::AsInner;

use libc::{self, c_int, c_void, ssize_t};

#[derive(Debug)]
pub struct FileDesc {
    fd: c_int,
}

fn max_len() -> usize {
    // The maximum read limit on most posix-like systems is `SSIZE_MAX`,
    // with the man page quoting that if the count of bytes to read is
    // greater than `SSIZE_MAX` the result is "unspecified".
    <ssize_t>::max_value() as usize
}

impl FileDesc {
    pub fn new(fd: c_int) -> FileDesc {
        FileDesc { fd: fd }
    }

    pub fn raw(&self) -> c_int {
        self.fd
    }

    /// Extracts the actual filedescriptor without closing it.
    pub fn into_raw(self) -> c_int {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::read(self.fd, buf.as_mut_ptr() as *mut c_void, cmp::min(buf.len(), max_len()))
        })?;
        Ok(ret as usize)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::readv(
                self.fd,
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), c_int::max_value() as usize) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        unsafe fn cvt_pread(
            fd: c_int,
            buf: *mut c_void,
            count: usize,
            offset: i64,
        ) -> io::Result<isize> {
            use libc::pread;
            cvt(pread(fd, buf, count, offset))
        }

        unsafe {
            cvt_pread(
                self.fd,
                buf.as_mut_ptr() as *mut c_void,
                cmp::min(buf.len(), max_len()),
                offset as i64,
            )
            .map(|n| n as usize)
        }
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::write(self.fd, buf.as_ptr() as *const c_void, cmp::min(buf.len(), max_len()))
        })?;
        Ok(ret as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            libc::writev(
                self.fd,
                bufs.as_ptr() as *const libc::iovec,
                cmp::min(bufs.len(), c_int::max_value() as usize) as c_int,
            )
        })?;
        Ok(ret as usize)
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        unsafe fn cvt_pwrite(
            fd: c_int,
            buf: *const c_void,
            count: usize,
            offset: i64,
        ) -> io::Result<isize> {
            use libc::pwrite;
            cvt(pwrite(fd, buf, count, offset))
        }

        unsafe {
            cvt_pwrite(
                self.fd,
                buf.as_ptr() as *const c_void,
                cmp::min(buf.len(), max_len()),
                offset as i64,
            )
            .map(|n| n as usize)
        }
    }

    pub fn get_cloexec(&self) -> io::Result<bool> {
        unsafe { Ok((cvt(libc::fcntl(self.fd, libc::F_GETFD))? & libc::FD_CLOEXEC) != 0) }
    }

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

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        unsafe {
            let v = nonblocking as c_int;
            cvt(libc::ioctl(self.fd, libc::FIONBIO, &v))?;
            Ok(())
        }
    }

    // refer to pxPipeDrv library documentation.
    // VxWorks uses fcntl to set O_NONBLOCK to the pipes
    pub fn set_nonblocking_pipe(&self, nonblocking: bool) -> io::Result<()> {
        unsafe {
            let mut flags = cvt(libc::fcntl(self.fd, libc::F_GETFL, 0))?;
            flags = if nonblocking { flags | libc::O_NONBLOCK } else { flags & !libc::O_NONBLOCK };
            cvt(libc::fcntl(self.fd, libc::F_SETFL, flags))?;
            Ok(())
        }
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        let fd = self.raw();
        match cvt(unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) }) {
            Ok(newfd) => Ok(FileDesc::new(newfd)),
            Err(e) => return Err(e),
        }
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
    fn drop(&mut self) {
        // Note that errors are ignored when closing a file descriptor. The
        // reason for this is that if an error occurs we don't actually know if
        // the file descriptor was closed or not, and if we retried (for
        // something like EINTR), we might close another valid file descriptor
        // (opened after we closed ours.
        let _ = unsafe { libc::close(self.fd) };
    }
}
