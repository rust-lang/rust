//! WASI-specific extensions to general I/O primitives

#![deny(unsafe_op_in_unsafe_fn)]
#![unstable(feature = "wasi_ext", issue = "none")]

use crate::fs;
use crate::io;
use crate::net;
use crate::sys;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// Raw file descriptors.
pub type RawFd = u32;

/// A trait to extract the raw WASI file descriptor from an underlying
/// object.
pub trait AsRawFd {
    /// Extracts the raw file descriptor.
    ///
    /// This method does **not** pass ownership of the raw file descriptor
    /// to the caller. The descriptor is only guaranteed to be valid while
    /// the original object has not yet been destroyed.
    fn as_raw_fd(&self) -> RawFd;
}

/// A trait to express the ability to construct an object from a raw file
/// descriptor.
pub trait FromRawFd {
    /// Constructs a new instance of `Self` from the given raw file
    /// descriptor.
    ///
    /// This function **consumes ownership** of the specified file
    /// descriptor. The returned object will take responsibility for closing
    /// it when the object goes out of scope.
    ///
    /// This function is also unsafe as the primitives currently returned
    /// have the contract that they are the sole owner of the file
    /// descriptor they are wrapping. Usage of this function could
    /// accidentally allow violating this contract which can cause memory
    /// unsafety in code that relies on it being true.
    unsafe fn from_raw_fd(fd: RawFd) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw file descriptor.
pub trait IntoRawFd {
    /// Consumes this object, returning the raw underlying file descriptor.
    ///
    /// This function **transfers ownership** of the underlying file descriptor
    /// to the caller. Callers are then the unique owners of the file descriptor
    /// and must close the descriptor once it's no longer needed.
    fn into_raw_fd(self) -> RawFd;
}

#[stable(feature = "raw_fd_reflexive_traits", since = "1.48.0")]
impl AsRawFd for RawFd {
    fn as_raw_fd(&self) -> RawFd {
        *self
    }
}
#[stable(feature = "raw_fd_reflexive_traits", since = "1.48.0")]
impl IntoRawFd for RawFd {
    fn into_raw_fd(self) -> RawFd {
        self
    }
}
#[stable(feature = "raw_fd_reflexive_traits", since = "1.48.0")]
impl FromRawFd for RawFd {
    unsafe fn from_raw_fd(fd: RawFd) -> RawFd {
        fd
    }
}

impl AsRawFd for net::TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().as_raw()
    }
}

impl FromRawFd for net::TcpStream {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpStream {
        net::TcpStream::from_inner(sys::net::TcpStream::from_inner(fd))
    }
}

impl IntoRawFd for net::TcpStream {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

impl AsRawFd for net::TcpListener {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().as_raw()
    }
}

impl FromRawFd for net::TcpListener {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpListener {
        net::TcpListener::from_inner(sys::net::TcpListener::from_inner(fd))
    }
}

impl IntoRawFd for net::TcpListener {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

impl AsRawFd for net::UdpSocket {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().as_raw()
    }
}

impl FromRawFd for net::UdpSocket {
    unsafe fn from_raw_fd(fd: RawFd) -> net::UdpSocket {
        net::UdpSocket::from_inner(sys::net::UdpSocket::from_inner(fd))
    }
}

impl IntoRawFd for net::UdpSocket {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

impl AsRawFd for fs::File {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().as_raw()
    }
}

impl FromRawFd for fs::File {
    unsafe fn from_raw_fd(fd: RawFd) -> fs::File {
        fs::File::from_inner(sys::fs::File::from_inner(fd))
    }
}

impl IntoRawFd for fs::File {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

impl AsRawFd for io::Stdin {
    fn as_raw_fd(&self) -> RawFd {
        sys::stdio::Stdin.as_raw_fd()
    }
}

impl AsRawFd for io::Stdout {
    fn as_raw_fd(&self) -> RawFd {
        sys::stdio::Stdout.as_raw_fd()
    }
}

impl AsRawFd for io::Stderr {
    fn as_raw_fd(&self) -> RawFd {
        sys::stdio::Stderr.as_raw_fd()
    }
}

impl<'a> AsRawFd for io::StdinLock<'a> {
    fn as_raw_fd(&self) -> RawFd {
        sys::stdio::Stdin.as_raw_fd()
    }
}

impl<'a> AsRawFd for io::StdoutLock<'a> {
    fn as_raw_fd(&self) -> RawFd {
        sys::stdio::Stdout.as_raw_fd()
    }
}

impl<'a> AsRawFd for io::StderrLock<'a> {
    fn as_raw_fd(&self) -> RawFd {
        sys::stdio::Stderr.as_raw_fd()
    }
}
