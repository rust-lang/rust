use super::*;

use crate::{
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd},
    process::Stdio,
    sys::{
        fd::FileDesc,
        pipe::{anon_pipe, AnonPipe},
    },
    sys_common::{FromInner, IntoInner},
};

#[inline]
pub(super) fn pipe() -> io::Result<(PipeReader, PipeWriter)> {
    anon_pipe().map(|(rx, tx)| (PipeReader(rx), PipeWriter(tx)))
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
        FileDesc::into_inner(AnonPipe::into_inner(pipe.0))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawFd for PipeReader {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(AnonPipe::from_raw_fd(raw_fd))
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
        FileDesc::into_inner(AnonPipe::into_inner(pipe.0))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawFd for PipeWriter {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self(AnonPipe::from_raw_fd(raw_fd))
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

fn convert_to_pipe(owned_fd: OwnedFd) -> AnonPipe {
    AnonPipe::from_inner(FileDesc::from_inner(OwnedFd::from(owned_fd)))
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<OwnedFd> for PipeReader {
    fn from(owned_fd: OwnedFd) -> Self {
        Self(convert_to_pipe(owned_fd))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<OwnedFd> for PipeWriter {
    fn from(owned_fd: OwnedFd) -> Self {
        Self(convert_to_pipe(owned_fd))
    }
}
