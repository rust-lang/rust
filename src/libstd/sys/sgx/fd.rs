use fortanix_sgx_abi::Fd;

use io;
use mem;
use sys::{AsInner, FromInner, IntoInner};
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
        usercalls::read(self.fd, buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        usercalls::write(self.fd, buf)
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
