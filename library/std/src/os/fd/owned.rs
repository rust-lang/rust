//! Owned and borrowed Unix-like file descriptors.

#![stable(feature = "io_safety", since = "1.63.0")]
#![deny(unsafe_op_in_unsafe_fn)]

use super::raw::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use crate::fmt;
use crate::fs;
use crate::marker::PhantomData;
use crate::mem::forget;
#[cfg(not(any(target_arch = "wasm32", target_env = "sgx")))]
use crate::sys::cvt;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// A borrowed file descriptor.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the file descriptor.
///
/// This uses `repr(transparent)` and has the representation of a host file
/// descriptor, so it can be used in FFI in places where a file descriptor is
/// passed as an argument, it is not captured or consumed, and it never has the
/// value `-1`.
///
/// This type's `.to_owned()` implementation returns another `BorrowedFd`
/// rather than an `OwnedFd`. It just makes a trivial copy of the raw file
/// descriptor, which is then borrowed under the same lifetime.
#[derive(Copy, Clone)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// libstd/os/raw/mod.rs assures me that every libstd-supported platform has a
// 32-bit c_int. Below is -2, in two's complement, but that only works out
// because c_int is 32 bits.
#[rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE)]
#[rustc_nonnull_optimization_guaranteed]
#[stable(feature = "io_safety", since = "1.63.0")]
pub struct BorrowedFd<'fd> {
    fd: RawFd,
    _phantom: PhantomData<&'fd OwnedFd>,
}

/// An owned file descriptor.
///
/// This closes the file descriptor on drop.
///
/// This uses `repr(transparent)` and has the representation of a host file
/// descriptor, so it can be used in FFI in places where a file descriptor is
/// passed as a consumed argument or returned as an owned value, and it never
/// has the value `-1`.
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// libstd/os/raw/mod.rs assures me that every libstd-supported platform has a
// 32-bit c_int. Below is -2, in two's complement, but that only works out
// because c_int is 32 bits.
#[rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE)]
#[rustc_nonnull_optimization_guaranteed]
#[stable(feature = "io_safety", since = "1.63.0")]
pub struct OwnedFd {
    fd: RawFd,
}

impl BorrowedFd<'_> {
    /// Return a `BorrowedFd` holding the given raw file descriptor.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `fd` must remain open for the duration of
    /// the returned `BorrowedFd`, and it must not have the value `-1`.
    #[inline]
    #[rustc_const_stable(feature = "io_safety", since = "1.63.0")]
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub const unsafe fn borrow_raw(fd: RawFd) -> Self {
        assert!(fd != u32::MAX as RawFd);
        // SAFETY: we just asserted that the value is in the valid range and isn't `-1` (the only value bigger than `0xFF_FF_FF_FE` unsigned)
        unsafe { Self { fd, _phantom: PhantomData } }
    }
}

impl OwnedFd {
    /// Creates a new `OwnedFd` instance that shares the same underlying file
    /// description as the existing `OwnedFd` instance.
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone(&self) -> crate::io::Result<Self> {
        self.as_fd().try_clone_to_owned()
    }
}

impl BorrowedFd<'_> {
    /// Creates a new `OwnedFd` instance that shares the same underlying file
    /// description as the existing `BorrowedFd` instance.
    #[cfg(not(target_arch = "wasm32"))]
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone_to_owned(&self) -> crate::io::Result<OwnedFd> {
        // We want to atomically duplicate this file descriptor and set the
        // CLOEXEC flag, and currently that's done via F_DUPFD_CLOEXEC. This
        // is a POSIX flag that was added to Linux in 2.6.24.
        #[cfg(not(target_os = "espidf"))]
        let cmd = libc::F_DUPFD_CLOEXEC;

        // For ESP-IDF, F_DUPFD is used instead, because the CLOEXEC semantics
        // will never be supported, as this is a bare metal framework with
        // no capabilities for multi-process execution.  While F_DUPFD is also
        // not supported yet, it might be (currently it returns ENOSYS).
        #[cfg(target_os = "espidf")]
        let cmd = libc::F_DUPFD;

        // Avoid using file descriptors below 3 as they are used for stdio
        let fd = cvt(unsafe { libc::fcntl(self.as_raw_fd(), cmd, 3) })?;
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Creates a new `OwnedFd` instance that shares the same underlying file
    /// description as the existing `BorrowedFd` instance.
    #[cfg(target_arch = "wasm32")]
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone_to_owned(&self) -> crate::io::Result<OwnedFd> {
        Err(crate::io::const_io_error!(
            crate::io::ErrorKind::Unsupported,
            "operation not supported on WASI yet",
        ))
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawFd for BorrowedFd<'_> {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawFd for OwnedFd {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl IntoRawFd for OwnedFd {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        let fd = self.fd;
        forget(self);
        fd
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl FromRawFd for OwnedFd {
    /// Constructs a new instance of `Self` from the given raw file descriptor.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `fd` must be open and suitable for assuming
    /// ownership. The resource must not require any cleanup other than `close`.
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        assert_ne!(fd, u32::MAX as RawFd);
        // SAFETY: we just asserted that the value is in the valid range and isn't `-1` (the only value bigger than `0xFF_FF_FF_FE` unsigned)
        unsafe { Self { fd } }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl Drop for OwnedFd {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Note that errors are ignored when closing a file descriptor. The
            // reason for this is that if an error occurs we don't actually know if
            // the file descriptor was closed or not, and if we retried (for
            // something like EINTR), we might close another valid file descriptor
            // opened after we closed ours.
            let _ = libc::close(self.fd);
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for BorrowedFd<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedFd").field("fd", &self.fd).finish()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for OwnedFd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedFd").field("fd", &self.fd).finish()
    }
}

/// A trait to borrow the file descriptor from an underlying object.
///
/// This is only available on unix platforms and must be imported in order to
/// call the method. Windows platforms have a corresponding `AsHandle` and
/// `AsSocket` set of traits.
#[stable(feature = "io_safety", since = "1.63.0")]
pub trait AsFd {
    /// Borrows the file descriptor.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::fs::File;
    /// # use std::io;
    /// # #[cfg(target_os = "wasi")]
    /// # use std::os::wasi::io::{AsFd, BorrowedFd};
    /// # #[cfg(unix)]
    /// # use std::os::unix::io::{AsFd, BorrowedFd};
    ///
    /// let mut f = File::open("foo.txt")?;
    /// # #[cfg(any(unix, target_os = "wasi"))]
    /// let borrowed_fd: BorrowedFd<'_> = f.as_fd();
    /// # Ok::<(), io::Error>(())
    /// ```
    #[stable(feature = "io_safety", since = "1.63.0")]
    fn as_fd(&self) -> BorrowedFd<'_>;
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsFd> AsFd for &T {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        T::as_fd(self)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsFd> AsFd for &mut T {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        T::as_fd(self)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for BorrowedFd<'_> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        *self
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for OwnedFd {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        // Safety: `OwnedFd` and `BorrowedFd` have the same validity
        // invariants, and the `BorrowdFd` is bounded by the lifetime
        // of `&self`.
        unsafe { BorrowedFd::borrow_raw(self.as_raw_fd()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for fs::File {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<fs::File> for OwnedFd {
    #[inline]
    fn from(file: fs::File) -> OwnedFd {
        file.into_inner().into_inner().into_inner()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for fs::File {
    #[inline]
    fn from(owned_fd: OwnedFd) -> Self {
        Self::from_inner(FromInner::from_inner(FromInner::from_inner(owned_fd)))
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for crate::net::TcpStream {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().socket().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::TcpStream> for OwnedFd {
    #[inline]
    fn from(tcp_stream: crate::net::TcpStream) -> OwnedFd {
        tcp_stream.into_inner().into_socket().into_inner().into_inner().into()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for crate::net::TcpStream {
    #[inline]
    fn from(owned_fd: OwnedFd) -> Self {
        Self::from_inner(FromInner::from_inner(FromInner::from_inner(FromInner::from_inner(
            owned_fd,
        ))))
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for crate::net::TcpListener {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().socket().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::TcpListener> for OwnedFd {
    #[inline]
    fn from(tcp_listener: crate::net::TcpListener) -> OwnedFd {
        tcp_listener.into_inner().into_socket().into_inner().into_inner().into()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for crate::net::TcpListener {
    #[inline]
    fn from(owned_fd: OwnedFd) -> Self {
        Self::from_inner(FromInner::from_inner(FromInner::from_inner(FromInner::from_inner(
            owned_fd,
        ))))
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for crate::net::UdpSocket {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().socket().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::UdpSocket> for OwnedFd {
    #[inline]
    fn from(udp_socket: crate::net::UdpSocket) -> OwnedFd {
        udp_socket.into_inner().into_socket().into_inner().into_inner().into()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for crate::net::UdpSocket {
    #[inline]
    fn from(owned_fd: OwnedFd) -> Self {
        Self::from_inner(FromInner::from_inner(FromInner::from_inner(FromInner::from_inner(
            owned_fd,
        ))))
    }
}

#[stable(feature = "asfd_ptrs", since = "1.64.0")]
/// This impl allows implementing traits that require `AsFd` on Arc.
/// ```
/// # #[cfg(any(unix, target_os = "wasi"))] mod group_cfg {
/// # #[cfg(target_os = "wasi")]
/// # use std::os::wasi::io::AsFd;
/// # #[cfg(unix)]
/// # use std::os::unix::io::AsFd;
/// use std::net::UdpSocket;
/// use std::sync::Arc;
///
/// trait MyTrait: AsFd {}
/// impl MyTrait for Arc<UdpSocket> {}
/// impl MyTrait for Box<UdpSocket> {}
/// # }
/// ```
impl<T: AsFd> AsFd for crate::sync::Arc<T> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        (**self).as_fd()
    }
}

#[stable(feature = "asfd_ptrs", since = "1.64.0")]
impl<T: AsFd> AsFd for Box<T> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        (**self).as_fd()
    }
}
