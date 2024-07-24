use crate::{
    io,
    os::windows::io::{
        AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle,
    },
    pipe::{PipeReader, PipeWriter},
    process::Stdio,
    sys::{handle::Handle, pipe::unnamed_anon_pipe},
    sys_common::{FromInner, IntoInner},
};

pub(crate) type AnonPipe = Handle;

#[inline]
pub(crate) fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    unnamed_anon_pipe().map(|(rx, wx)| (rx.into_inner(), wx.into_inner()))
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsHandle for PipeReader {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.0.as_handle()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsRawHandle for PipeReader {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.as_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawHandle for PipeReader {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        Self(Handle::from_raw_handle(raw_handle))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl IntoRawHandle for PipeReader {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeReader> for OwnedHandle {
    fn from(pipe: PipeReader) -> Self {
        Handle::into_inner(pipe.0)
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeReader> for Stdio {
    fn from(pipe: PipeReader) -> Self {
        Self::from(OwnedHandle::from(pipe))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsHandle for PipeWriter {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.0.as_handle()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsRawHandle for PipeWriter {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.as_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawHandle for PipeWriter {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        Self(Handle::from_raw_handle(raw_handle))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl IntoRawHandle for PipeWriter {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for OwnedHandle {
    fn from(pipe: PipeWriter) -> Self {
        Handle::into_inner(pipe.0)
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        Self::from(OwnedHandle::from(pipe))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<OwnedHandle> for PipeReader {
    fn from(owned_handle: OwnedHandle) -> Self {
        Self(Handle::from_inner(owned_handle))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<OwnedHandle> for PipeWriter {
    fn from(owned_handle: OwnedHandle) -> Self {
        Self(Handle::from_inner(owned_handle))
    }
}
