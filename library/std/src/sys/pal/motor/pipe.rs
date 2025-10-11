use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::fd::FileDesc;
use crate::sys::map_motor_error;
use crate::sys_common::{FromInner, IntoInner};

#[derive(Debug)]
pub struct AnonPipe(FileDesc);

impl From<moto_rt::RtFd> for AnonPipe {
    fn from(rt_fd: moto_rt::RtFd) -> AnonPipe {
        unsafe { AnonPipe::from_raw_fd(rt_fd) }
    }
}

impl AnonPipe {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        moto_rt::fs::read(self.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|b| self.read(b), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        moto_rt::fs::write(self.as_raw_fd(), buf).map_err(map_motor_error)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|b| self.write(b), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut temp_vec = Vec::new();
        let mut size = 0_usize;
        loop {
            temp_vec.resize(256, 0_u8);
            match self.read(&mut temp_vec[..]) {
                Ok(sz) => {
                    if sz == 0 {
                        return Ok(size);
                    }
                    size += sz;
                    temp_vec.truncate(sz);
                    buf.append(&mut temp_vec);
                }
                Err(err) => {
                    if size != 0 {
                        return Ok(size);
                    } else {
                        return Err(err);
                    }
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

impl FromRawFd for AnonPipe {
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        let desc = FileDesc::from_raw_fd(fd);
        Self(desc)
    }
}

impl IntoRawFd for AnonPipe {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}

impl AsFd for AnonPipe {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl IntoInner<OwnedFd> for AnonPipe {
    fn into_inner(self) -> OwnedFd {
        self.0.into_inner()
    }
}

impl IntoInner<FileDesc> for AnonPipe {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

impl FromInner<OwnedFd> for AnonPipe {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self(FileDesc::from_inner(owned_fd))
    }
}

pub fn read2(_p1: AnonPipe, _v1: &mut Vec<u8>, _p2: AnonPipe, _v2: &mut Vec<u8>) -> io::Result<()> {
    Err(io::Error::from_raw_os_error(moto_rt::E_NOT_IMPLEMENTED.into()))
}

#[inline]
pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    Err(io::Error::UNSUPPORTED_PLATFORM)
}
