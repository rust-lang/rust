use fortanix_sgx_abi::Fd;

use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem;
use crate::sys::{AsInner, FromInner, IntoInner};
use super::abi::usercalls;

#[derive(Debug)]
pub struct FileDesc {
    fd: Fd,
}

impl FileDesc {
    pub fn new(fd: Fd) -> FileDesc {
        FileDesc { fd: fd }
    }

    pub fn raw(&self) -> Fd { self.fd }

    /// Extracts the actual filedescriptor without closing it.
    pub fn into_raw(self) -> Fd {
        let fd = self.fd;
        mem::forget(self);
        fd
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        usercalls::read(self.fd, &mut [IoSliceMut::new(buf)])
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        usercalls::read(self.fd, bufs)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        usercalls::write(self.fd, &[IoSlice::new(buf)])
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        usercalls::write(self.fd, bufs)
    }

    pub fn flush(&self) -> io::Result<()> {
        usercalls::flush(self.fd)
    }
}

impl AsInner<Fd> for FileDesc {
    fn as_inner(&self) -> &Fd { &self.fd }
}

impl IntoInner<Fd> for FileDesc {
    fn into_inner(self) -> Fd {
        let fd = self.fd;
        mem::forget(self);
        fd
    }
}

impl FromInner<Fd> for FileDesc {
    fn from_inner(fd: Fd) -> FileDesc {
        FileDesc { fd }
    }
}

impl Drop for FileDesc {
    fn drop(&mut self) {
        usercalls::close(self.fd)
    }
}
