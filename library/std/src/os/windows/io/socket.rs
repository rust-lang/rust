//! Owned and borrowed OS sockets.

#![unstable(feature = "io_safety", issue = "87074")]

use super::raw::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::mem;
use crate::mem::forget;
use crate::sys;
use crate::sys::c;
use crate::sys::cvt;

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
    socket: RawSocket,
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
    socket: RawSocket,
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
    pub unsafe fn borrow_raw_socket(socket: RawSocket) -> Self {
        debug_assert_ne!(socket, c::INVALID_SOCKET as RawSocket);
        Self { socket, _phantom: PhantomData }
    }
}

impl OwnedSocket {
    /// Creates a new `OwnedSocket` instance that shares the same underlying socket
    /// as the existing `OwnedSocket` instance.
    pub fn try_clone(&self) -> io::Result<Self> {
        let mut info = unsafe { mem::zeroed::<c::WSAPROTOCOL_INFO>() };
        let result = unsafe {
            c::WSADuplicateSocketW(self.as_raw_socket(), c::GetCurrentProcessId(), &mut info)
        };
        sys::net::cvt(result)?;
        let socket = unsafe {
            c::WSASocketW(
                info.iAddressFamily,
                info.iSocketType,
                info.iProtocol,
                &mut info,
                0,
                c::WSA_FLAG_OVERLAPPED | c::WSA_FLAG_NO_HANDLE_INHERIT,
            )
        };

        if socket != c::INVALID_SOCKET {
            unsafe { Ok(OwnedSocket::from_raw_socket(socket)) }
        } else {
            let error = unsafe { c::WSAGetLastError() };

            if error != c::WSAEPROTOTYPE && error != c::WSAEINVAL {
                return Err(io::Error::from_raw_os_error(error));
            }

            let socket = unsafe {
                c::WSASocketW(
                    info.iAddressFamily,
                    info.iSocketType,
                    info.iProtocol,
                    &mut info,
                    0,
                    c::WSA_FLAG_OVERLAPPED,
                )
            };

            if socket == c::INVALID_SOCKET {
                return Err(last_error());
            }

            unsafe {
                let socket = OwnedSocket::from_raw_socket(socket);
                socket.set_no_inherit()?;
                Ok(socket)
            }
        }
    }

    #[cfg(not(target_vendor = "uwp"))]
    pub(crate) fn set_no_inherit(&self) -> io::Result<()> {
        cvt(unsafe {
            c::SetHandleInformation(self.as_raw_socket() as c::HANDLE, c::HANDLE_FLAG_INHERIT, 0)
        })
        .map(drop)
    }

    #[cfg(target_vendor = "uwp")]
    pub(crate) fn set_no_inherit(&self) -> io::Result<()> {
        Err(io::const_io_error!(io::ErrorKind::Unsupported, "Unavailable on UWP"))
    }
}

/// Returns the last error from the Windows socket interface.
fn last_error() -> io::Error {
    io::Error::from_raw_os_error(unsafe { c::WSAGetLastError() })
}

impl AsRawSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.socket
    }
}

impl AsRawSocket for OwnedSocket {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.socket
    }
}

impl IntoRawSocket for OwnedSocket {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        let socket = self.socket;
        forget(self);
        socket
    }
}

impl FromRawSocket for OwnedSocket {
    /// Constructs a new instance of `Self` from the given raw socket.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `socket` must be open and suitable for
    /// assuming ownership. The resource must not require cleanup other than
    /// `closesocket`.
    #[inline]
    unsafe fn from_raw_socket(socket: RawSocket) -> Self {
        debug_assert_ne!(socket, c::INVALID_SOCKET as RawSocket);
        Self { socket }
    }
}

impl Drop for OwnedSocket {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = c::closesocket(self.socket);
        }
    }
}

impl fmt::Debug for BorrowedSocket<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedSocket").field("socket", &self.socket).finish()
    }
}

impl fmt::Debug for OwnedSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedSocket").field("socket", &self.socket).finish()
    }
}

/// A trait to borrow the socket from an underlying object.
#[unstable(feature = "io_safety", issue = "87074")]
pub trait AsSocket {
    /// Borrows the socket.
    fn as_socket(&self) -> BorrowedSocket<'_>;
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<T: AsSocket> AsSocket for &T {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        T::as_socket(self)
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<T: AsSocket> AsSocket for &mut T {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        T::as_socket(self)
    }
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
        // Safety: `OwnedSocket` and `BorrowedSocket` have the same validity
        // invariants, and the `BorrowdSocket` is bounded by the lifetime
        // of `&self`.
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
