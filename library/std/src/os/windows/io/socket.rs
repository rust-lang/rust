//! Owned and borrowed OS sockets.

#![stable(feature = "io_safety", since = "1.63.0")]

use super::raw::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::mem;
use crate::mem::forget;
use crate::sys;
use crate::sys::c;
#[cfg(not(target_vendor = "uwp"))]
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
///
/// This type's `.to_owned()` implementation returns another `BorrowedSocket`
/// rather than an `OwnedSocket`. It just makes a trivial copy of the raw
/// socket, which is then borrowed under the same lifetime.
#[derive(Copy, Clone)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// This is -2, in two's complement. -1 is `INVALID_SOCKET`.
#[cfg_attr(target_pointer_width = "32", rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE))]
#[cfg_attr(
    target_pointer_width = "64",
    rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FF_FF_FF_FF_FE)
)]
#[rustc_nonnull_optimization_guaranteed]
#[stable(feature = "io_safety", since = "1.63.0")]
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
#[rustc_nonnull_optimization_guaranteed]
#[stable(feature = "io_safety", since = "1.63.0")]
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
    #[rustc_const_stable(feature = "io_safety", since = "1.63.0")]
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub const unsafe fn borrow_raw(socket: RawSocket) -> Self {
        assert!(socket != c::INVALID_SOCKET as RawSocket);
        Self { socket, _phantom: PhantomData }
    }
}

impl OwnedSocket {
    /// Creates a new `OwnedSocket` instance that shares the same underlying socket
    /// as the existing `OwnedSocket` instance.
    #[stable(feature = "io_safety", since = "1.63.0")]
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

    // FIXME(strict_provenance_magic): we defined RawSocket to be a u64 ;-;
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

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.socket
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawSocket for OwnedSocket {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.socket
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl IntoRawSocket for OwnedSocket {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        let socket = self.socket;
        forget(self);
        socket
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl FromRawSocket for OwnedSocket {
    #[inline]
    unsafe fn from_raw_socket(socket: RawSocket) -> Self {
        debug_assert_ne!(socket, c::INVALID_SOCKET as RawSocket);
        Self { socket }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl Drop for OwnedSocket {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = c::closesocket(self.socket);
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for BorrowedSocket<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedSocket").field("socket", &self.socket).finish()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for OwnedSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedSocket").field("socket", &self.socket).finish()
    }
}

/// A trait to borrow the socket from an underlying object.
#[stable(feature = "io_safety", since = "1.63.0")]
pub trait AsSocket {
    /// Borrows the socket.
    #[stable(feature = "io_safety", since = "1.63.0")]
    fn as_socket(&self) -> BorrowedSocket<'_>;
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsSocket> AsSocket for &T {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        T::as_socket(self)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsSocket> AsSocket for &mut T {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        T::as_socket(self)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        *self
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for OwnedSocket {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        // Safety: `OwnedSocket` and `BorrowedSocket` have the same validity
        // invariants, and the `BorrowdSocket` is bounded by the lifetime
        // of `&self`.
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for crate::net::TcpStream {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::TcpStream> for OwnedSocket {
    #[inline]
    fn from(tcp_stream: crate::net::TcpStream) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(tcp_stream.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedSocket> for crate::net::TcpStream {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for crate::net::TcpListener {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::TcpListener> for OwnedSocket {
    #[inline]
    fn from(tcp_listener: crate::net::TcpListener) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(tcp_listener.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedSocket> for crate::net::TcpListener {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for crate::net::UdpSocket {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::UdpSocket> for OwnedSocket {
    #[inline]
    fn from(udp_socket: crate::net::UdpSocket) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(udp_socket.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedSocket> for crate::net::UdpSocket {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}
