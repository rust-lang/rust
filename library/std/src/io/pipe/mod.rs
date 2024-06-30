use crate::{io, process::Stdio, sys::pipe::AnonPipe};

/// Create annoymous pipe that is close-on-exec and blocking.
#[inline]
pub fn pipe() -> io::Result<(PipeReader, PipeWriter)> {
    cfg_if::cfg_if! {
        if #[cfg(unix)] {
            unix::pipe()
        } else {
            windows::pipe()
        }
    }
}

/// Read end of the annoymous pipe.
#[derive(Debug)]
pub struct PipeReader(AnonPipe);

/// Write end of the annoymous pipe.
#[derive(Debug)]
pub struct PipeWriter(AnonPipe);

impl PipeReader {
    /// Create a new [`PipeReader`] instance that shares the same underlying file description.
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0.try_clone().map(Self)
    }
}

impl PipeWriter {
    /// Create a new [`PipeWriter`] instance that shares the same underlying file description.
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0.try_clone().map(Self)
    }
}

macro_rules! forward_io_read_traits {
    ($name:ty) => {
        impl io::Read for $name {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.0.read(buf)
            }
            fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
                self.0.read_vectored(bufs)
            }
            fn is_read_vectored(&self) -> bool {
                self.0.is_read_vectored()
            }
            fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
                self.0.read_to_end(buf)
            }
            fn read_buf(&mut self, buf: io::BorrowedCursor<'_>) -> io::Result<()> {
                self.0.read_buf(buf)
            }
        }
    };
}
forward_io_read_traits!(PipeReader);
forward_io_read_traits!(&PipeReader);

macro_rules! forward_io_write_traits {
    ($name:ty) => {
        impl io::Write for $name {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }

            fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
                self.0.write_vectored(bufs)
            }
            fn is_write_vectored(&self) -> bool {
                self.0.is_write_vectored()
            }
        }
    };
}
forward_io_write_traits!(PipeWriter);
forward_io_write_traits!(&PipeWriter);

#[cfg(unix)]
mod unix {
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

    macro_rules! impl_traits {
        ($name:ty) => {
            impl AsFd for $name {
                fn as_fd(&self) -> BorrowedFd<'_> {
                    self.0.as_fd()
                }
            }
            impl AsRawFd for $name {
                fn as_raw_fd(&self) -> RawFd {
                    self.0.as_raw_fd()
                }
            }
            impl From<$name> for OwnedFd {
                fn from(pipe: $name) -> Self {
                    FileDesc::into_inner(AnonPipe::into_inner(pipe.0))
                }
            }
            impl FromRawFd for $name {
                unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
                    Self(AnonPipe::from_raw_fd(raw_fd))
                }
            }
            impl IntoRawFd for $name {
                fn into_raw_fd(self) -> RawFd {
                    self.0.into_raw_fd()
                }
            }
            impl From<$name> for Stdio {
                fn from(pipe: $name) -> Self {
                    Self::from(OwnedFd::from(pipe))
                }
            }
        };
    }
    impl_traits!(PipeReader);
    impl_traits!(PipeWriter);

    fn convert_to_pipe(owned_fd: OwnedFd) -> io::Result<AnonPipe> {
        let file = File::from(owned_fd);
        if file.metadata()?.file_type().is_fifo() {
            Ok(AnonPipe::from_inner(FileDesc::from_inner(OwnedFd::from(file))))
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "Not a pipe"))
        }
    }

    enum AccessMode {
        Readable,
        Writable,
    }

    fn check_access_mode(pipe: AnonPipe, expected_access_mode: AccessMode) -> io::Result<AnonPipe> {
        let ret = unsafe { libc::fcntl(pipe.as_raw_fd(), libc::F_GETFL) };
        let access_mode = ret & libc::O_ACCMODE;
        let expected_access_mode_str = match expected_access_mode {
            AccessMode::Readable => "readable",
            AccessMode::Writable => "writable",
        };
        let expected_access_mode = match expected_access_mode {
            AccessMode::Readable => libc::O_RDONLY,
            AccessMode::Writable => libc::O_WRONLY,
        };

        if ret == -1 {
            Err(io::Error::last_os_error())
        } else if access_mode == libc::O_RDWR && access_mode == expected_access_mode {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Pipe {} is not {}", pipe.as_raw_fd(), expected_access_mode_str),
            ))
        } else {
            Ok(pipe)
        }
    }

    impl TryFrom<OwnedFd> for PipeReader {
        type Error = io::Error;

        fn try_from(owned_fd: OwnedFd) -> Result<Self, Self::Error> {
            convert_to_pipe(owned_fd)
                .and_then(|pipe| check_access_mode(pipe, AccessMode::Readable))
                .map(Self)
        }
    }

    impl TryFrom<OwnedFd> for PipeWriter {
        type Error = io::Error;

        fn try_from(owned_fd: OwnedFd) -> Result<Self, Self::Error> {
            convert_to_pipe(owned_fd)
                .and_then(|pipe| check_access_mode(pipe, AccessMode::Writable))
                .map(Self)
        }
    }
}

#[cfg(windows)]
mod windows {
    use super::*;

    use crate::{
        os::windows::io::{
            AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle,
            RawHandle,
        },
        sys::{
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
            impl AsHandle for $name {
                fn as_handle(&self) -> BorrowedHandle<'_> {
                    self.0.handle().as_handle()
                }
            }
            impl AsRawHandle for $name {
                fn as_raw_handle(&self) -> RawHandle {
                    self.0.handle().as_raw_handle()
                }
            }

            impl FromRawHandle for $name {
                unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
                    Self(AnonPipe::from_inner(Handle::from_raw_handle(raw_handle)))
                }
            }
            impl IntoRawHandle for $name {
                fn into_raw_handle(self) -> RawHandle {
                    self.0.into_handle().into_raw_handle()
                }
            }

            impl From<$name> for OwnedHandle {
                fn from(pipe: $name) -> Self {
                    Handle::into_inner(AnonPipe::into_inner(pipe.0))
                }
            }
            impl From<$name> for Stdio {
                fn from(pipe: $name) -> Self {
                    Self::from(OwnedHandle::from(pipe))
                }
            }
        };
    }
    impl_traits!(PipeReader);
    impl_traits!(PipeWriter);

    fn owned_handle_to_anon_pipe(owned_handle: OwnedHandle) -> AnonPipe {
        AnonPipe::from_inner(Handle::from_inner(owned_handle))
    }

    impl TryFrom<OwnedHandle> for PipeReader {
        type Error = io::Error;

        fn try_from(owned_handle: OwnedHandle) -> Result<Self, Self::Error> {
            Ok(Self(owned_handle_to_anon_pipe(owned_handle)))
        }
    }

    impl TryFrom<OwnedHandle> for PipeWriter {
        type Error = io::Error;

        fn try_from(owned_handle: OwnedHandle) -> Result<Self, Self::Error> {
            Ok(Self(owned_handle_to_anon_pipe(owned_handle)))
        }
    }
}
