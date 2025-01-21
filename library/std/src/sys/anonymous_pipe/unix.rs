use crate::io::{self, PipeReader, PipeWriter};
use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::process::Stdio;
use crate::sys::fd::FileDesc;
use crate::sys::pipe::anon_pipe;
use crate::sys_common::{FromInner, IntoInner};

pub type AnonPipe = FileDesc;

#[inline]
pub fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    anon_pipe().map(|(rx, wx)| (rx.into_inner(), wx.into_inner()))
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsFd for PipeReader {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsRawFd for PipeReader {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeReader> for OwnedFd {
    fn from(pipe: PipeReader) -> Self {
        FileDesc::into_inner(pipe.0)
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl FromRawFd for PipeReader {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self(FileDesc::from_raw_fd(raw_fd)) }
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl IntoRawFd for PipeReader {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeReader> for Stdio {
    fn from(pipe: PipeReader) -> Self {
        Self::from(OwnedFd::from(pipe))
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsFd for PipeWriter {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsRawFd for PipeWriter {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeWriter> for OwnedFd {
    fn from(pipe: PipeWriter) -> Self {
        FileDesc::into_inner(pipe.0)
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl FromRawFd for PipeWriter {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self(FileDesc::from_raw_fd(raw_fd)) }
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl IntoRawFd for PipeWriter {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        Self::from(OwnedFd::from(pipe))
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<OwnedFd> for PipeReader {
    fn from(owned_fd: OwnedFd) -> Self {
        Self(FileDesc::from_inner(owned_fd))
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<OwnedFd> for PipeWriter {
    fn from(owned_fd: OwnedFd) -> Self {
        Self(FileDesc::from_inner(owned_fd))
    }
}
