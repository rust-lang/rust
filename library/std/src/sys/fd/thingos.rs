#![unstable(reason = "not public", issue = "none", feature = "fd")]

use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, Read};
use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::pal::common::{
    F_GETFL, F_SETFL, O_NONBLOCK, SYS_DUP, SYS_FCNTL, SYS_READ, SYS_WRITE, cvt, raw_syscall6,
    syscall3,
};
use crate::sys::{AsInner, FromInner, IntoInner};

#[derive(Debug)]
pub struct FileDesc(OwnedFd);

impl FileDesc {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let ret =
            unsafe { raw_syscall6(SYS_READ, self.as_raw_fd() as u64, buf.as_mut_ptr() as u64, buf.len() as u64, 0, 0, 0) };
        cvt(ret).map(|n| n as usize)
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        io::default_read_vectored(|b| self.read(b), bufs)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut me = self;
        (&mut me).read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let ret =
            unsafe { raw_syscall6(SYS_WRITE, self.as_raw_fd() as u64, buf.as_ptr() as u64, buf.len() as u64, 0, 0, 0) };
        cvt(ret).map(|n| n as usize)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let flags = cvt(unsafe { syscall3(SYS_FCNTL, self.as_raw_fd() as u64, F_GETFL, 0) })?;
        let new_flags = if nonblocking {
            flags as u64 | O_NONBLOCK
        } else {
            flags as u64 & !O_NONBLOCK
        };
        cvt(unsafe { syscall3(SYS_FCNTL, self.as_raw_fd() as u64, F_SETFL, new_flags) })?;
        Ok(())
    }

    pub fn duplicate(&self) -> io::Result<FileDesc> {
        let new_fd =
            cvt(unsafe { raw_syscall6(SYS_DUP, self.as_raw_fd() as u64, 0, 0, 0, 0, 0) })?;
        // SAFETY: `new_fd` is a valid file descriptor returned by the kernel.
        unsafe { Ok(Self::from_raw_fd(new_fd as RawFd)) }
    }

    #[inline]
    pub fn try_clone(&self) -> io::Result<Self> {
        self.duplicate()
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
        // SAFETY: caller guarantees raw_fd is valid and owned.
        unsafe { Self(FromRawFd::from_raw_fd(raw_fd)) }
    }
}
