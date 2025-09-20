use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};

pub struct Pipe(!);

#[inline]
pub fn pipe() -> io::Result<(Pipe, Pipe)> {
    Err(io::Error::UNSUPPORTED_PLATFORM)
}

impl Pipe {
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

impl fmt::Debug for Pipe {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

#[cfg(any(unix, target_os = "hermit", target_os = "wasi"))]
mod unix_traits {
    use super::ChildPipe;
    use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
    use crate::sys_common::FromInner;

    impl AsRawFd for ChildPipe {
        #[inline]
        fn as_raw_fd(&self) -> RawFd {
            self.0
        }
    }

    impl AsFd for ChildPipe {
        fn as_fd(&self) -> BorrowedFd<'_> {
            self.0
        }
    }

    impl IntoRawFd for ChildPipe {
        fn into_raw_fd(self) -> RawFd {
            self.0
        }
    }

    impl FromRawFd for ChildPipe {
        unsafe fn from_raw_fd(_: RawFd) -> Self {
            panic!("creating pipe on this platform is unsupported!")
        }
    }

    impl FromInner<OwnedFd> for ChildPipe {
        fn from_inner(_: OwnedFd) -> Self {
            panic!("creating pipe on this platform is unsupported!")
        }
    }
}
