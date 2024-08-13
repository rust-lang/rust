#![unstable(reason = "not public", issue = "none", feature = "fd")]

use super::hermit_abi;
use crate::cmp;
use crate::io::{self, IoSlice, IoSliceMut, Read};
use crate::os::hermit::io::{FromRawFd, OwnedFd, RawFd, *};
use crate::sys::{cvt, unsupported};
use crate::sys_common::{AsInner, FromInner, IntoInner};

const fn max_iov() -> usize {
    hermit_abi::IOV_MAX
}

#[derive(Debug)]
pub struct FileDesc {
    fd: OwnedFd,
}

impl FileDesc {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let result =
            cvt(unsafe { hermit_abi::read(self.fd.as_raw_fd(), buf.as_mut_ptr(), buf.len()) })?;
        Ok(result as usize)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            hermit_abi::readv(
                self.as_raw_fd(),
                bufs.as_mut_ptr() as *mut hermit_abi::iovec as *const hermit_abi::iovec,
                cmp::min(bufs.len(), max_iov()),
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

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let result =
            cvt(unsafe { hermit_abi::write(self.fd.as_raw_fd(), buf.as_ptr(), buf.len()) })?;
        Ok(result as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let ret = cvt(unsafe {
            hermit_abi::writev(
                self.as_raw_fd(),
                bufs.as_ptr() as *const hermit_abi::iovec,
                cmp::min(bufs.len(), max_iov()),
            )
        })?;
        Ok(ret as usize)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
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

    pub fn fstat(&self, stat: *mut hermit_abi::stat) -> io::Result<()> {
        cvt(unsafe { hermit_abi::fstat(self.fd.as_raw_fd(), stat) })?;
        Ok(())
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
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        Self { fd }
    }
}

impl AsInner<OwnedFd> for FileDesc {
    #[inline]
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
