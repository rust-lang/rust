use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::os::cvt;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

#[derive(Debug)]
pub struct WasiFd {
    fd: OwnedFd,
}

impl WasiFd {
    pub fn read(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let bufs = crate::sys::io::IoSliceMut::as_libc_slice(bufs);
        unsafe {
            let n = cvt(libc::readv(self.as_raw_fd(), bufs.as_ptr(), bufs.len() as libc::c_int))?;
            Ok(n as usize)
        }
    }

    pub fn read_buf(&self, mut buf: BorrowedCursor<'_>) -> io::Result<()> {
        unsafe {
            let amt = cvt(libc::read(
                self.as_raw_fd(),
                buf.as_mut().as_mut_ptr().cast(),
                buf.as_mut().len(),
            ))?;
            buf.advance_unchecked(amt as usize);
            Ok(())
        }
    }

    pub fn write(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let bufs = crate::sys::io::IoSlice::as_libc_slice(bufs);
        unsafe {
            let n = cvt(libc::writev(self.as_raw_fd(), bufs.as_ptr(), bufs.len() as libc::c_int))?;
            Ok(n as usize)
        }
    }
}

impl AsInner<OwnedFd> for WasiFd {
    #[inline]
    fn as_inner(&self) -> &OwnedFd {
        &self.fd
    }
}

impl AsInnerMut<OwnedFd> for WasiFd {
    #[inline]
    fn as_inner_mut(&mut self) -> &mut OwnedFd {
        &mut self.fd
    }
}

impl IntoInner<OwnedFd> for WasiFd {
    fn into_inner(self) -> OwnedFd {
        self.fd
    }
}

impl FromInner<OwnedFd> for WasiFd {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self { fd: owned_fd }
    }
}

impl AsFd for WasiFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl AsRawFd for WasiFd {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl IntoRawFd for WasiFd {
    fn into_raw_fd(self) -> RawFd {
        self.fd.into_raw_fd()
    }
}

impl FromRawFd for WasiFd {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self { fd: FromRawFd::from_raw_fd(raw_fd) } }
    }
}
