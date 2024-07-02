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

macro_rules! impl_traits {
    ($name:ty) => {
        #[unstable(feature = "anonymous_pipe", issue = "127154")]
        impl AsHandle for $name {
            fn as_handle(&self) -> BorrowedHandle<'_> {
                self.0.handle().as_handle()
            }
        }
        #[unstable(feature = "anonymous_pipe", issue = "127154")]
        impl AsRawHandle for $name {
            fn as_raw_handle(&self) -> RawHandle {
                self.0.handle().as_raw_handle()
            }
        }

        #[unstable(feature = "anonymous_pipe", issue = "127154")]
        impl FromRawHandle for $name {
            unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
                Self(AnonPipe::from_inner(Handle::from_raw_handle(raw_handle)))
            }
        }
        #[unstable(feature = "anonymous_pipe", issue = "127154")]
        impl IntoRawHandle for $name {
            fn into_raw_handle(self) -> RawHandle {
                self.0.into_handle().into_raw_handle()
            }
        }

        #[unstable(feature = "anonymous_pipe", issue = "127154")]
        impl From<$name> for OwnedHandle {
            fn from(pipe: $name) -> Self {
                Handle::into_inner(AnonPipe::into_inner(pipe.0))
            }
        }
        #[unstable(feature = "anonymous_pipe", issue = "127154")]
        impl From<$name> for Stdio {
            fn from(pipe: $name) -> Self {
                Self::from(OwnedHandle::from(pipe))
            }
        }
    };
}
impl_traits!(PipeReader);
impl_traits!(PipeWriter);

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
