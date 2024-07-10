use super::*;

use crate::{
    os::windows::io::{
        AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle,
    },
    sys::{
        c::{GetFileType, FILE_TYPE_PIPE},
        handle::Handle,
        pipe::{anon_pipe, AnonPipe, Pipes},
    },
    sys_common::{FromInner, IntoInner},
};

#[inline]
pub(super) fn pipe() -> io::Result<(PipeReader, PipeWriter)> {
    anon_pipe(true, false).map(|Pipes { ours, theirs }| (PipeReader(ours), PipeWriter(theirs)))
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsHandle for PipeReader {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.0.handle().as_handle()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsRawHandle for PipeReader {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.handle().as_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawHandle for PipeReader {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        Self(AnonPipe::from_inner(Handle::from_raw_handle(raw_handle)))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl IntoRawHandle for PipeReader {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_handle().into_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeReader> for OwnedHandle {
    fn from(pipe: PipeReader) -> Self {
        Handle::into_inner(AnonPipe::into_inner(pipe.0))
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
        self.0.handle().as_handle()
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl AsRawHandle for PipeWriter {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.handle().as_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl FromRawHandle for PipeWriter {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        Self(AnonPipe::from_inner(Handle::from_raw_handle(raw_handle)))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl IntoRawHandle for PipeWriter {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_handle().into_raw_handle()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for OwnedHandle {
    fn from(pipe: PipeWriter) -> Self {
        Handle::into_inner(AnonPipe::into_inner(pipe.0))
    }
}
#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        Self::from(OwnedHandle::from(pipe))
    }
}

fn convert_to_pipe(owned_handle: OwnedHandle) -> io::Result<AnonPipe> {
    if unsafe { GetFileType(owned_handle.as_raw_handle()) } == FILE_TYPE_PIPE {
        Ok(AnonPipe::from_inner(Handle::from_inner(owned_handle)))
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "Not a pipe"))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl TryFrom<OwnedHandle> for PipeReader {
    type Error = io::Error;

    fn try_from(owned_handle: OwnedHandle) -> Result<Self, Self::Error> {
        convert_to_pipe(owned_handle).map(Self)
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl TryFrom<OwnedHandle> for PipeWriter {
    type Error = io::Error;

    fn try_from(owned_handle: OwnedHandle) -> Result<Self, Self::Error> {
        convert_to_pipe(owned_handle).map(Self)
    }
}
