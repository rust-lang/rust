//! Owned and borrowed OS sockets.

#![unstable(feature = "io_safety", issue = "87074")]

use super::raw::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::fmt;
use crate::marker::PhantomData;
use crate::mem::forget;
use crate::sys::c;

/// A borrowed socket.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the socket.
///
/// This uses `repr(transparent)` and has the representation of a host socket,
/// so it can be used in FFI in places where a socket is passed as an argument,
/// it is not captured or consumed, and it never has the value
/// `INVALID_SOCKET`.
#[derive(Copy, Clone)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// This is -2, in two's complement. -1 is `INVALID_SOCKET`.
#[cfg_attr(target_pointer_width = "32", rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE))]
#[cfg_attr(
    target_pointer_width = "64",
    rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FF_FF_FF_FF_FE)
)]
#[unstable(feature = "io_safety", issue = "87074")]
pub struct BorrowedSocket<'socket> {
    raw: RawSocket,
    _phantom: PhantomData<&'socket OwnedSocket>,
}

/// An owned socket.
///
/// This closes the socket on drop.
///
/// This uses `repr(transparent)` and has the representation of a host socket,
/// so it can be used in FFI in places where a socket is passed as a consumed
/// argument or returned as an owned value, and it never has the value
/// `INVALID_SOCKET`.
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// This is -2, in two's complement. -1 is `INVALID_SOCKET`.
#[cfg_attr(target_pointer_width = "32", rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE))]
#[cfg_attr(
    target_pointer_width = "64",
    rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FF_FF_FF_FF_FE)
)]
#[unstable(feature = "io_safety", issue = "87074")]
pub struct OwnedSocket {
    raw: RawSocket,
}

impl BorrowedSocket<'_> {
    /// Return a `BorrowedSocket` holding the given raw socket.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `raw` must remain open for the duration of
    /// the returned `BorrowedSocket`, and it must not have the value
    /// `INVALID_SOCKET`.
    #[inline]
    #[unstable(feature = "io_safety", issue = "87074")]
    pub unsafe fn borrow_raw_socket(raw: RawSocket) -> Self {
        debug_assert_ne!(raw, c::INVALID_SOCKET as RawSocket);
        Self { raw, _phantom: PhantomData }
    }
}

impl AsRawSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.raw
    }
}

impl AsRawSocket for OwnedSocket {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.raw
    }
}

impl IntoRawSocket for OwnedSocket {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        let raw = self.raw;
        forget(self);
        raw
    }
}

impl FromRawSocket for OwnedSocket {
    /// Constructs a new instance of `Self` from the given raw socket.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `raw` must be open and suitable for assuming
    /// ownership. The resource must not require cleanup other than `closesocket`.
    #[inline]
    unsafe fn from_raw_socket(raw: RawSocket) -> Self {
        debug_assert_ne!(raw, c::INVALID_SOCKET as RawSocket);
        Self { raw }
    }
}

impl Drop for OwnedSocket {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = c::closesocket(self.raw);
        }
    }
}

impl fmt::Debug for BorrowedSocket<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedSocket").field("socket", &self.raw).finish()
    }
}

impl fmt::Debug for OwnedSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedSocket").field("socket", &self.raw).finish()
    }
}

/// A trait to borrow the socket from an underlying object.
#[unstable(feature = "io_safety", issue = "87074")]
pub trait AsSocket {
    /// Borrows the socket.
    fn as_socket(&self) -> BorrowedSocket<'_>;
}

impl AsSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        *self
    }
}

impl AsSocket for OwnedSocket {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw_socket(self.as_raw_socket()) }
    }
}

impl AsSocket for crate::net::TcpStream {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw_socket(self.as_raw_socket()) }
    }
}

impl From<crate::net::TcpStream> for OwnedSocket {
    #[inline]
    fn from(tcp_stream: crate::net::TcpStream) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(tcp_stream.into_raw_socket()) }
    }
}

impl From<OwnedSocket> for crate::net::TcpStream {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}

impl AsSocket for crate::net::TcpListener {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw_socket(self.as_raw_socket()) }
    }
}

impl From<crate::net::TcpListener> for OwnedSocket {
    #[inline]
    fn from(tcp_listener: crate::net::TcpListener) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(tcp_listener.into_raw_socket()) }
    }
}

impl From<OwnedSocket> for crate::net::TcpListener {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}

impl AsSocket for crate::net::UdpSocket {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw_socket(self.as_raw_socket()) }
    }
}

impl From<crate::net::UdpSocket> for OwnedSocket {
    #[inline]
    fn from(udp_socket: crate::net::UdpSocket) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(udp_socket.into_raw_socket()) }
    }
}

impl From<OwnedSocket> for crate::net::UdpSocket {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}
