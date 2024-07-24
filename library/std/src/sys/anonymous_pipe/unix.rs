use crate::{
    io,
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd},
    pipe::{PipeReader, PipeWriter},
    process::Stdio,
    sys::{fd::FileDesc, pipe::anon_pipe},
    sys_common::{FromInner, IntoInner},
};

pub(crate) type AnonPipe = FileDesc;

#[inline]
pub(crate) fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    anon_pipe().map(|(rx, wx)| (rx.into_inner(), wx.into_inner()))
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsFd for PipeReader {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsRawFd for PipeReader {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeReader> for OwnedFd {
    fn from(pipe: PipeReader) -> Self {
        FileDesc::into_inner(pipe.0)
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawFd for PipeReader {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(FileDesc::from_raw_fd(raw_fd))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl IntoRawFd for PipeReader {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeReader> for Stdio {
    fn from(pipe: PipeReader) -> Self {
        Self::from(OwnedFd::from(pipe))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsFd for PipeWriter {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsRawFd for PipeWriter {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for OwnedFd {
    fn from(pipe: PipeWriter) -> Self {
        FileDesc::into_inner(pipe.0)
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawFd for PipeWriter {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(FileDesc::from_raw_fd(raw_fd))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl IntoRawFd for PipeWriter {
    fn into_raw_fd(self) -> RawFd {
        self.0.into_raw_fd()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        Self::from(OwnedFd::from(pipe))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<OwnedFd> for PipeReader {
    fn from(owned_fd: OwnedFd) -> Self {
        Self(FileDesc::from_inner(owned_fd))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<OwnedFd> for PipeWriter {
    fn from(owned_fd: OwnedFd) -> Self {
        Self(FileDesc::from_inner(owned_fd))
    }
}
