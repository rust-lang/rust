use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::sys_common::{FromInner, IntoInner};

pub struct AnonPipe(!);

impl fmt::Debug for AnonPipe {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl AnonPipe {
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0
    }

    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn read_buf(&self, _buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.0
    }

    pub fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_read_vectored(&self) -> bool {
        self.0
    }

    pub fn read_to_end(&self, _buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0
    }

    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_write_vectored(&self) -> bool {
        self.0
    }

    pub fn diverge(&self) -> ! {
        self.0
    }
}

pub fn read2(p1: AnonPipe, _v1: &mut Vec<u8>, _p2: AnonPipe, _v2: &mut Vec<u8>) -> io::Result<()> {
    match p1.0 {}
}

impl FromInner<!> for AnonPipe {
    fn from_inner(inner: !) -> Self {
        inner
    }
}

impl IntoInner<!> for AnonPipe {
    fn into_inner(self) -> ! {
        self.0
    }
}

#[cfg(any(unix, target_os = "hermit", target_os = "wasi"))]
mod unix_traits {
    use super::AnonPipe;
    use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
    use crate::sys_common::FromInner;

    impl AsRawFd for AnonPipe {
        #[inline]
        fn as_raw_fd(&self) -> RawFd {
            self.0
        }
    }

    impl AsFd for AnonPipe {
        fn as_fd(&self) -> BorrowedFd<'_> {
            self.0
        }
    }

    impl IntoRawFd for AnonPipe {
        fn into_raw_fd(self) -> RawFd {
            self.0
        }
    }

    impl FromRawFd for AnonPipe {
        unsafe fn from_raw_fd(_: RawFd) -> Self {
            panic!("creating pipe on this platform is unsupported!")
        }
    }

    impl FromInner<OwnedFd> for AnonPipe {
        fn from_inner(_: OwnedFd) -> Self {
            panic!("creating pipe on this platform is unsupported!")
        }
    }
}
