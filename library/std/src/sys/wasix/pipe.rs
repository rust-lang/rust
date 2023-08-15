use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys_common::IntoInner;
use crate::sys::err2io;
use crate::io::Read;

pub use crate::sys::{cvt, cvt_r};

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe(FileDesc);

pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let (fd1, fd2) = unsafe {
        wasi::fd_pipe().map_err(err2io)?
    };
    let fd1 = fd1 as RawFd;
    let fd2 = fd2 as RawFd;

    unsafe {
        Ok((AnonPipe(FileDesc::from_raw_fd(fd1)), AnonPipe(FileDesc::from_raw_fd(fd2))))
    }
}

impl AnonPipe {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(&mut [IoSliceMut::new(buf)])
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        (&self.0).read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(buf)])
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }
}

impl IntoInner<FileDesc> for AnonPipe {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

pub fn read2(p1: AnonPipe, v1: &mut Vec<u8>, p2: AnonPipe, v2: &mut Vec<u8>) -> io::Result<()> {
    // Set both pipes into nonblocking mode as we're gonna be reading from both
    // in the `select` loop below, and we wouldn't want one to block the other!
    let p1 = p1.into_inner();
    let p2 = p2.into_inner();
    p1.set_nonblocking(true)?;
    p2.set_nonblocking(true)?;

    let mut fds: [libc::pollfd; 2] = unsafe { mem::zeroed() };
    fds[0].fd = p1.as_raw_fd();
    fds[0].events = libc::POLLIN;
    fds[1].fd = p2.as_raw_fd();
    fds[1].events = libc::POLLIN;
    loop {
        // wait for either pipe to become readable using `poll`
        cvt_r(|| unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) })?;

        if fds[0].revents != 0 && read(&p1, v1)? {
            p2.set_nonblocking(false)?;
            return (&p2).read_to_end(v2).map(drop);
        }
        if fds[1].revents != 0 && read(&p2, v2)? {
            p1.set_nonblocking(false)?;
            return (&p1).read_to_end(v1).map(drop);
        }
    }

    // Read as much as we can from each pipe, ignoring EWOULDBLOCK or
    // EAGAIN. If we hit EOF, then this will happen because the underlying
    // reader will return Ok(0), in which case we'll see `Ok` ourselves. In
    // this case we flip the other fd back into blocking mode and read
    // whatever's leftover on that file descriptor.
    fn read(mut fd: &FileDesc, dst: &mut Vec<u8>) -> Result<bool, io::Error> {
        match fd.read_to_end(dst) {
            Ok(_) => Ok(true),
            Err(e) => {
                if e.raw_os_error() == Some(libc::EWOULDBLOCK)
                    || e.raw_os_error() == Some(libc::EAGAIN)
                {
                    Ok(false)
                } else {
                    Err(e)
                }
            }
        }
    }
}

impl AsRawFd for AnonPipe {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl AsFd for AnonPipe {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl IntoRawFd for AnonPipe {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl FromRawFd for AnonPipe {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(FromRawFd::from_raw_fd(raw_fd))
    }
}
