#![unstable(reason = "not public", issue = "none", feature = "fd")]

use super::hermit_abi;
use crate::cmp;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, Read, SeekFrom};
use crate::os::hermit::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
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

    pub fn read_buf(&self, mut buf: BorrowedCursor<'_>) -> io::Result<()> {
        // SAFETY: The `read` syscall does not read from the buffer, so it is
        // safe to use `&mut [MaybeUninit<u8>]`.
        let result = cvt(unsafe {
            hermit_abi::read(
                self.fd.as_raw_fd(),
                buf.as_mut().as_mut_ptr() as *mut u8,
                buf.capacity(),
            )
        })?;
        // SAFETY: Exactly `result` bytes have been filled.
        unsafe { buf.advance_unchecked(result as usize) };
        Ok(())
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

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, too large values will end up as
            // negative which will cause an error in `lseek`.
            SeekFrom::Start(off) => (hermit_abi::SEEK_SET, off as i64),
            SeekFrom::End(off) => (hermit_abi::SEEK_END, off),
            SeekFrom::Current(off) => (hermit_abi::SEEK_CUR, off),
        };
        let n = cvt(unsafe { hermit_abi::lseek(self.as_raw_fd(), pos as isize, whence) })?;
        Ok(n as u64)
    }

    pub fn tell(&self) -> io::Result<u64> {
        self.seek(SeekFrom::Current(0))
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
