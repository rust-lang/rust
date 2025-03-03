use crate::io::{self, PipeReader, PipeWriter};
use crate::os::windows::io::{
    AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle,
};
use crate::process::Stdio;
use crate::ptr;
use crate::sys::c;
use crate::sys::handle::Handle;
use crate::sys_common::{FromInner, IntoInner};

pub type AnonPipe = Handle;

pub fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let mut read_pipe = c::INVALID_HANDLE_VALUE;
    let mut write_pipe = c::INVALID_HANDLE_VALUE;

    let ret = unsafe { c::CreatePipe(&mut read_pipe, &mut write_pipe, ptr::null_mut(), 0) };

    if ret == 0 {
        Err(io::Error::last_os_error())
    } else {
        unsafe { Ok((Handle::from_raw_handle(read_pipe), Handle::from_raw_handle(write_pipe))) }
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsHandle for PipeReader {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.0.as_handle()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsRawHandle for PipeReader {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.as_raw_handle()
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl FromRawHandle for PipeReader {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        unsafe { Self(Handle::from_raw_handle(raw_handle)) }
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl IntoRawHandle for PipeReader {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_raw_handle()
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeReader> for OwnedHandle {
    fn from(pipe: PipeReader) -> Self {
        Handle::into_inner(pipe.0)
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeReader> for Stdio {
    fn from(pipe: PipeReader) -> Self {
        Self::from(OwnedHandle::from(pipe))
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsHandle for PipeWriter {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.0.as_handle()
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl AsRawHandle for PipeWriter {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.as_raw_handle()
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl FromRawHandle for PipeWriter {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        unsafe { Self(Handle::from_raw_handle(raw_handle)) }
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl IntoRawHandle for PipeWriter {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_raw_handle()
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeWriter> for OwnedHandle {
    fn from(pipe: PipeWriter) -> Self {
        Handle::into_inner(pipe.0)
    }
}
#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        Self::from(OwnedHandle::from(pipe))
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<OwnedHandle> for PipeReader {
    fn from(owned_handle: OwnedHandle) -> Self {
        Self(Handle::from_inner(owned_handle))
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<OwnedHandle> for PipeWriter {
    fn from(owned_handle: OwnedHandle) -> Self {
        Self(Handle::from_inner(owned_handle))
    }
}
