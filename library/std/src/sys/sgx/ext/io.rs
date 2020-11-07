//! SGX-specific extensions to general I/O primitives
//!
//! SGX file descriptors behave differently from Unix file descriptors. See the
//! description of [`TryIntoRawFd`] for more details.
#![unstable(feature = "sgx_platform", issue = "56975")]

use crate::net;
pub use crate::sys::abi::usercalls::raw::Fd as RawFd;
use crate::sys::{self, AsInner, FromInner, IntoInner, TryIntoInner};

/// A trait to extract the raw SGX file descriptor from an underlying
/// object.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub trait AsRawFd {
    /// Extracts the raw file descriptor.
    ///
    /// This method does **not** pass ownership of the raw file descriptor
    /// to the caller. The descriptor is only guaranteed to be valid while
    /// the original object has not yet been destroyed.
    #[unstable(feature = "sgx_platform", issue = "56975")]
    fn as_raw_fd(&self) -> RawFd;
}

/// A trait to express the ability to construct an object from a raw file
/// descriptor.
#[unstable(feature = "sgx_platform", issue = "56975")]
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
    #[unstable(feature = "sgx_platform", issue = "56975")]
    unsafe fn from_raw_fd(fd: RawFd) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw file descriptor.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub trait TryIntoRawFd: Sized {
    /// Consumes this object, returning the raw underlying file descriptor, if
    /// this object is not cloned.
    ///
    /// This function **transfers ownership** of the underlying file descriptor
    /// to the caller. Callers are then the unique owners of the file descriptor
    /// and must close the descriptor once it's no longer needed.
    ///
    /// Unlike other platforms, on SGX, the file descriptor is shared between
    /// all clones of an object. To avoid race conditions, this function will
    /// only return `Ok` when called on the final clone.
    #[unstable(feature = "sgx_platform", issue = "56975")]
    fn try_into_raw_fd(self) -> Result<RawFd, Self>;
}

impl AsRawFd for net::TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        *self.as_inner().as_inner().as_inner().as_inner()
    }
}

impl AsRawFd for net::TcpListener {
    fn as_raw_fd(&self) -> RawFd {
        *self.as_inner().as_inner().as_inner().as_inner()
    }
}

impl FromRawFd for net::TcpStream {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpStream {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let socket = sys::net::Socket::from_inner(fd);
        net::TcpStream::from_inner(sys::net::TcpStream::from_inner((socket, None)))
    }
}

impl FromRawFd for net::TcpListener {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpListener {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let socket = sys::net::Socket::from_inner(fd);
        net::TcpListener::from_inner(sys::net::TcpListener::from_inner(socket))
    }
}

impl TryIntoRawFd for net::TcpStream {
    fn try_into_raw_fd(self) -> Result<RawFd, Self> {
        let (socket, peer_addr) = self.into_inner().into_inner();
        match socket.try_into_inner() {
            Ok(fd) => Ok(fd.into_inner()),
            Err(socket) => {
                let sys = sys::net::TcpStream::from_inner((socket, peer_addr));
                Err(net::TcpStream::from_inner(sys))
            }
        }
    }
}

impl TryIntoRawFd for net::TcpListener {
    fn try_into_raw_fd(self) -> Result<RawFd, Self> {
        match self.into_inner().into_inner().try_into_inner() {
            Ok(fd) => Ok(fd.into_inner()),
            Err(socket) => {
                let sys = sys::net::TcpListener::from_inner(socket);
                Err(net::TcpListener::from_inner(sys))
            }
        }
    }
}
