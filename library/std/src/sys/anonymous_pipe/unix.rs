use super::*;

use crate::{
    fs::File,
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd},
        unix::fs::FileTypeExt,
    },
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

fn convert_to_pipe(owned_fd: OwnedFd) -> io::Result<AnonPipe> {
    let file = File::from(owned_fd);
    if file.metadata()?.file_type().is_fifo() {
        Ok(AnonPipe::from_inner(FileDesc::from_inner(OwnedFd::from(file))))
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "Not a pipe"))
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl TryFrom<OwnedFd> for PipeReader {
    type Error = io::Error;

    fn try_from(owned_fd: OwnedFd) -> Result<Self, Self::Error> {
        convert_to_pipe(owned_fd)
            .and_then(|pipe| {
                if pipe.as_file_desc().get_access_mode()?.is_readable() {
                    Ok(pipe)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Pipe {} is not readable", pipe.as_raw_fd()),
                    ))
                }
            })
            .map(Self)
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl TryFrom<OwnedFd> for PipeWriter {
    type Error = io::Error;

    fn try_from(owned_fd: OwnedFd) -> Result<Self, Self::Error> {
        convert_to_pipe(owned_fd)
            .and_then(|pipe| {
                if pipe.as_file_desc().get_access_mode()?.is_writable() {
                    Ok(pipe)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Pipe {} is not writable", pipe.as_raw_fd()),
                    ))
                }
            })
            .map(Self)
    }
}
