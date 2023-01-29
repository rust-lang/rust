#![unstable(reason = "not public", issue = "none", feature = "fd")]

use crate::io::{self, Read};
use crate::os::hermit::io::{FromRawFd, OwnedFd, RawFd};
use crate::sys::cvt;
use crate::sys::hermit::abi;
use crate::sys::unsupported;
use crate::sys_common::{AsInner, FromInner, IntoInner};

use crate::os::hermit::io::*;

#[derive(Debug)]
pub struct FileDesc {
    fd: OwnedFd,
}

impl FileDesc {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let result = cvt(unsafe { abi::read(self.fd.as_raw_fd(), buf.as_mut_ptr(), buf.len()) })?;
        Ok(result as usize)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let result = cvt(unsafe { abi::write(self.fd.as_raw_fd(), buf.as_ptr(), buf.len()) })?;
        Ok(result as usize)
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        self.duplicate_path(&[])
    }
    pub fn duplicate_path(&self, _path: &[u8]) -> io::Result<FileDesc> {
        unsupported()
    }

    pub fn nonblocking(&self) -> io::Result<bool> {
        Ok(false)
    }

    pub fn set_cloexec(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> {
        unsupported()
    }
}

impl<'a> Read for &'a FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }
}

impl IntoInner<OwnedFd> for FileDesc {
    fn into_inner(self) -> OwnedFd {
        self.fd
    }
}

impl FromInner<OwnedFd> for FileDesc {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self { fd: owned_fd }
    }
}

impl FromRawFd for FileDesc {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self { fd: FromRawFd::from_raw_fd(raw_fd) }
    }
}

impl AsInner<OwnedFd> for FileDesc {
    fn as_inner(&self) -> &OwnedFd {
        &self.fd
    }
}

impl AsFd for FileDesc {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl AsRawFd for FileDesc {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl IntoRawFd for FileDesc {
    fn into_raw_fd(self) -> RawFd {
        self.fd.into_raw_fd()
    }
}
